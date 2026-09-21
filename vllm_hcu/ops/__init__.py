# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from . import rotary_embedding
from . import silu_and_mul
from . import sigmoid_mul
from . import gemma_rms_norm
from . import rms_norm
from . import rms_norm_gated
# Registers torch.ops.vllm.hcu_rms_norm_gated_strided_z (port of d095594).
from . import hcu_rms_norm_gated_strided_z_op  # noqa: F401
