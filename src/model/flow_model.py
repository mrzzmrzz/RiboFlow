import torch
import torch.nn as nn
from einops import rearrange

from src.model.auxiliary_net import DistModule, MolNet
from src.model.edge_feature_net import EdgeFeatureNet
from src.model.ipa_net import IPANet
from src.model.msaformer import CrossAttention
from src.model.node_feature_net import NodeFeatureNet


# NUM_TOKENS = MASK_TOKEN_INDEX = X
num_torsions = 8


class FlowModel(nn.Module):
    def __init__(self, flow_model_conf):
        super(FlowModel, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.flow_model_conf = flow_model_conf
        self.node_feature_net = NodeFeatureNet(flow_model_conf.node_conf)
        self.edge_feature_net = EdgeFeatureNet(flow_model_conf.edge_conf)
        self.ipanet = IPANet(flow_model_conf.ipa_conf)
        self.node_embed_dim = self.flow_model_conf.node_embed_dim
        self.edge_embed_dim = self.flow_model_conf.edge_embed_dim
        self.condition_generation = self.flow_model_conf.guide_by_condition
        self.embed_aatype = flow_model_conf.node_conf.embed_aatype
        if self.embed_aatype:
            self.aatype_pred_net = nn.Sequential(
                nn.Linear(self.node_embed_dim, self.node_embed_dim),
                nn.ReLU(),
                nn.Linear(self.node_embed_dim, self.node_embed_dim),
                nn.ReLU(),
                nn.Linear(self.node_embed_dim, flow_model_conf.num_aa_types),
            )
        if self.condition_generation:
            self.mol_embedding_layer = MolNet(flow_model_conf.mol_conf)
            self.edge_dist_embedder = DistModule(flow_model_conf.dist_conf)
            self.bb_lig_fusion = CrossAttention(
                query_input_dim=self.node_embed_dim,
                key_input_dim=self.node_embed_dim,
                output_dim=self.node_embed_dim,
            )

    def forward(self, input_feats):
        flow_mask = input_feats["flow_mask"].type(torch.float32)
        bb_mask = input_feats["res_mask"].type(torch.float32)
        num_batch, num_res = flow_mask.shape
        is_na_residue_mask = input_feats["res_mask"].type(torch.bool)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]
        n_batch, n_res = bb_mask.shape
        init_bb_node_embed = self.node_feature_net(
            r3_t=rearrange(input_feats["r3_t"], "B -> B 1"),
            so3_t=rearrange(input_feats["so3_t"], "B -> B 1"),
            aa_t=rearrange(input_feats["aa_t"], "B -> B 1"),
            res_mask=bb_mask,
            flow_mask=flow_mask,
            pos=input_feats["res_idx"],
            aatypes=input_feats["aatypes_t"],
            sc_aatypes=input_feats["sc_aatypes"],
        )
        init_bb_edge_embed = self.edge_feature_net(
            node_features=init_bb_node_embed,
            trans=input_feats["trans_t"],
            sc_trans=input_feats["sc_trans"],
            edge_mask=edge_mask,
            flow_mask=flow_mask,
        )
        if self.condition_generation:
            lig_mask = input_feats["ligand_mask"]
            _, n_lig = lig_mask.shape
            lig_edge_mask = rearrange(lig_mask, "B L -> B L 1") * rearrange(
                lig_mask, "B L -> B 1 L"
            )
            lig_init_node_embed, _ = self.mol_embedding_layer(
                ligand_atom=input_feats["ligand_feat"],
                ligand_pos=input_feats["ligand_pos"],
                edge_mask=lig_edge_mask,
            )
            lig_node_embed = lig_init_node_embed[:, :n_lig, :] * rearrange(
                lig_mask, "B L -> B L 1"
            )
            bb_lig_rep, _ = self.bb_lig_fusion(
                query_input=bb_node_embed,
                key_input=lig_node_embed,
                value_input=lig_node_embed,
                query_input_mask=bb_mask,
                key_input_mask=lig_mask,
            )

            bb_node_embed = (bb_node_embed + bb_lig_rep) * rearrange(bb_mask, "B N -> B N 1")
            with torch.no_grad():
                model_out = self.ipanet(bb_node_embed, bb_edge_embed, input_feats)
            updated_trans_t = model_out["pred_trans"]
            updated_rotmats_t = model_out["pred_rotmats"]
            updated_torsions_t = model_out["pred_torsions"].reshape(
                num_batch, num_res, num_torsions * 2
            )
            bb_ligand_mask = torch.cat([bb_mask, lig_mask], dim=-1)
            bb_lig_edge = self.edge_dist_embedder(
                trans_t=updated_trans_t,
                rotmats_t=updated_rotmats_t,
                torsions_t=updated_torsions_t,
                lig_pos=input_feats["ligand_pos"],
                bb_mask=torch.ones_like(is_na_residue_mask),
                bb_ligand_mask=bb_ligand_mask,
            )
            bb_edge_embed = (bb_edge_embed + bb_lig_edge[:, :n_res, :n_res, :]) * rearrange(
                edge_mask, "B N1 N2 -> B N1 N2 1"
            )
        model_out = self.ipanet(bb_node_embed, bb_edge_embed, input_feats)
        if self.embed_aatype:
            node_embed = model_out["node_embed"]
            pred_logits = self.aatype_pred_net(node_embed) * rearrange(bb_mask, "B N -> B N 1")
            pred_aatypes = torch.argmax(pred_logits, dim=-1)
            if self.flow_model_conf.num_aa_types == NUM_TOKENS + 1:
                pred_logits_wo_mask = pred_logits.clone()
                pred_logits_wo_mask[:, :, MASK_TOKEN_INDEX] = -1e9
                pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
        else:
            pred_aatypes = input_feats["aatypes_t"].long()
            pred_logits = nn.functional.one_hot(
                pred_aatypes, num_classes=self.flow_model_conf.num_aa_types
            ).float()

        pred_out = {
            "pred_logits": pred_logits,
            "pred_aatypes": pred_aatypes,
            "pred_trans": model_out["pred_trans"],
            "pred_rotmats": model_out["pred_rotmats"],
            "pred_torsions": model_out["pred_torsions"],
        }

        return pred_out
