# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import UNet2DConditionLoadersMixin
from ..utils import BaseOutput, logging
from .activations import get_activation
from .attention_processor import AttentionProcessor, AttnProcessor
from .embeddings import (
    GaussianFourierProjection,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from .modeling_utils import ModelMixin
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UpBlock2D,
    get_down_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass(eq=False)
class EditLatentSpaceModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,   
            act_fn: str = "silu",
            in_channels: int = 4,   
            out_channels: int = 4,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),   
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            time_cond_proj_dim: Optional[int] = None,
            
            time_embedding_act_fn: Optional[str] = None,
            time_embedding_type: str = "positional",
            time_embedding_dim: Optional[int] = None,
            timestep_post_act: Optional[str] = None
            ):
        super().__init__()
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)
        self.time_map = nn.Linear(time_embed_dim, out_channels)

        self.conv_in = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        self.conv_out = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        
        self.group_norm = torch.nn.GroupNorm(in_channels, in_channels)
        self.swish = nn.SiLU()
    
    def forward(self, 
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                timestep_cond: Optional[torch.Tensor] = None,
                ):
        if not torch.is_tensor(timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timestep.expand(sample.shape[0])

        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)

        t_emb = self.time_embedding(t_emb, timestep_cond)

        sample = self.conv_in(self.swish(sample)) #+ self.swish(self.time_map(t_emb))[:, :, None, None]
        sample = self.conv_out(self.swish(self.group_norm(sample)))
        return sample
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value
