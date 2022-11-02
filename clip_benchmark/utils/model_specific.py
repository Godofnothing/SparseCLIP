import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module, Parameter
from typing import Optional, Tuple
from torch.nn.modules.activation import MultiheadAttention


__all__ = [
    "LayeredMultiheadAttention",
    "fix_attention_layer"
]


class LayeredMultiheadAttention(Module):

    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        dropout=0.0, 
        bias=True, 
        add_bias_kv=False, 
        add_zero_attn=False,
        kdim=None, 
        vdim=None, 
        batch_first=False, 
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayeredMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False: 
            # TODO merge bias in nn.Linear in future releases
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(embed_dim, self.kdim, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(embed_dim, self.vdim, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(3 * embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.scale = self.head_dim ** -0.5

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj.weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(LayeredMultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj.bias is not None and query.dtype != self.in_proj.bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj.bias ({self.in_proj.bias.dtype}) don't match"
        elif self.in_proj.weight is not None and query.dtype != self.in_proj.weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj.weight ({self.in_proj.weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj.weight,
                self.in_proj.bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj.weight,
                    self.in_proj.bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights
                )
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        N, B = query.shape[:2]
        if not self._qkv_same_embed_dim:
            query = self.q_proj(query).reshape(N, B, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            key = self.k_proj(key).reshape(N, B, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            value = self.v_proj(value).reshape(N, B, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            qkv = self.in_proj(query).reshape(
                N, B, 3, self.num_heads, self.head_dim).permute(2, 1, 3, 0, 4)
            query, key, value = qkv.unbind(0)  
        
        attn_output_weights = (query @ key.transpose(-2, -1)) * self.scale
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        if self.dropout > 0:
          attn_output_weights = F.dropout(attn_output_weights, self.dropout)

        # value = B, num_heads, L, head_dim
        attn_output = (attn_output_weights @ value).permute(2, 0, 1, 3).reshape(N, B, -1)
        attn_output = self.out_proj(attn_output)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


@torch.no_grad()
def fix_attention_layer(model: nn.Module):
    for module_name, module in model.named_modules():
        if isinstance(module, MultiheadAttention):
            bias = module.in_proj_bias is not None
            add_bias_kv = module.bias_k is not None

            attention = LayeredMultiheadAttention(
                module.embed_dim,
                module.num_heads,
                module.dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=module.add_zero_attn,
                kdim=module.kdim,
                vdim=module.vdim,
                batch_first=module.batch_first
            )

            # copy weights and biases
            with torch.no_grad():
                for proj_name in ['in', 'q', 'k', 'v']:
                    proj_weight = module.state_dict().get(f"{proj_name}_proj_weight")
                    if proj_weight is not None:
                        getattr(attention, f"{proj_name}_proj").weight.data = proj_weight
                if bias:
                    if not module._qkv_same_embed_dim:
                      q_bias, k_bias, v_bias = module.in_proj_bias.chunk(3)
                      attention.q_proj.data = q_bias
                      attention.k_proj.data = k_bias
                      attention.v_proj.data = v_bias
                    else:
                      attention.in_proj.bias.data = module.in_proj_bias
                if add_bias_kv:
                  attention.bias_k.data = module.bias_k
                  attention.bias_v.data = module.bias_v
                if module.out_proj.weight is not None:
                    attention.out_proj.weight.data = module.out_proj.weight.data
                if module.out_proj.bias is not None:
                    attention.out_proj.bias.data = module.out_proj.bias.data
            # get parent module
            parent_name, name = module_name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, name, attention)
    return model
     