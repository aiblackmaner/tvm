# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin, no-else-return
"""Conv3D operators"""
import tvm
from tvm import te

from .pad import pad
from .util import get_pad_tuple3d
from ..util import simplify, get_const_tuple
from .winograd_util import winograd_transform_matrices


def conv3d_ncdhw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Conv3D operator in NCDHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    Filter : tvm.te.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of three ints
        Stride size, or [strid_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_channel, in_depth, in_height, in_width = Input.shape
    num_filter, channel, kernel_d, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_front, pad_top, pad_left]
    pad_after = [0, 0, pad_back, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel), name="rc")
    rz = te.reduce_axis((0, kernel_d), name="rz")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    return te.compute(
        (batch, out_channel, out_depth, out_height, out_width),
        lambda nn, ff, zz, yy, xx: te.sum(
            temp[
                nn,
                rc,
                zz * stride_d + rz * dilation_d,
                yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w,
            ].astype(out_dtype)
            * Filter[ff, rc, rz, ry, rx].astype(out_dtype),
            axis=[rc, rz, ry, rx],
        ),
        tag="conv3d_ncdhw",
    )


def conv3d_ndhwc(Input, Filter, stride, padding, dilation, out_dtype="float32"):
    """Convolution operator in NDHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_depth, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        5-D with shape [filter_depth, filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of three ints
        Stride size, or [stride_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_depth, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3

    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_depth, in_height, in_width, in_channel = Input.shape
    kernel_d, kernel_h, kernel_w, channel, num_filter = Filter.shape
    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_front, pad_top, pad_left, 0]
    pad_after = [0, pad_back, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rd = te.reduce_axis((0, kernel_d), name="rd")
    rh = te.reduce_axis((0, kernel_h), name="rh")
    rw = te.reduce_axis((0, kernel_w), name="rw")
    rc = te.reduce_axis((0, in_channel), name="rc")
    Output = te.compute(
        (batch, out_depth, out_height, out_width, out_channel),
        lambda nn, dd, hh, ww, cc: te.sum(
            PaddedInput[
                nn,
                dd * stride_d + rd * dilation_d,
                hh * stride_h + rh * dilation_h,
                ww * stride_w + rw * dilation_w,
                rc,
            ].astype(out_dtype)
            * Filter[rd, rh, rw, rc, cc].astype(out_dtype),
            axis=[rd, rh, rw, rc],
        ),
        name="Conv3dOutput",
        tag="conv3d_ndhwc",
    )
    return Output
# modify
def conv3d_NCDHWc(data, kernel, strides, padding, dilation, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype

    assert isinstance(dilation, int) or len(dilation) == 3
    if isinstance(dilation, int):
        dilation_d, dilation_h, dilation_w = (dilation, dilation, dilation)
    else:
        dilation_d, dilation_h, dilation_w = dilation

    DSTR, HSTR, WSTR = strides
    batch_size, ic_chunk, in_depth, in_height, ic_bn, in_width = get_const_tuple(data.shape)
    oc_chunk, ic_chunk, kernel_depth, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
    in_channel = ic_chunk * ic_bn
    num_filter = oc_chunk * oc_bn
    dilated_kernel_d = (kernel_depth - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w)
    )

    pad_d = pad_front + pad_back
    pad_h = pad_top + pad_down
    pad_w = pad_left + pad_right

    pad_depth = in_depth + pad_d
    pad_height = in_height + pad_h
    pad_width = in_width + pad_w

    out_depth = simplify((in_depth + pad_d - dilated_kernel_d) // DSTR + 1)
    out_height = simplify((in_height + pad_h - dilated_kernel_h) // HSTR + 1)
    out_width = simplify((in_width + pad_w - dilated_kernel_w) // WSTR + 1)
    # pack data
    DOPAD = pad_d != 0 or pad_h != 0 or pad_w != 0
    if DOPAD:
        data_pad = pad(
            data,
            (0, 0, pad_front, pad_top, 0, pad_left),
            (0, 0, pad_back, pad_down, 0, pad_right),
            name="data_pad",
        )
    else:
        data_pad = data

    # fetch schedule
    # ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    # convolution
    oshape = (batch_size, num_filter // oc_bn, out_depth, out_height, out_width, oc_bn)

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")
    kd = te.reduce_axis((0, kernel_depth), name="kd")
    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    conv = te.compute(
        oshape,
        lambda n, oc_chunk, od, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                idxdiv(ic, ic_bn),
                od * DSTR + kd * dilation_d,
                oh * HSTR + kh * dilation_h,
                idxmod(ic, ic_bn),
                ow * WSTR + kw * dilation_w,
            ].astype(out_dtype)
            * kernel[
                oc_chunk, idxdiv(ic, ic_bn), kd, kh, kw, idxmod(ic, ic_bn), oc_block
            ].astype(out_dtype),
            axis=[ic, kd, kh, kw],
        ),
        name="conv3d_NCDHWc",
        tag="conv3d_NCDHWc",
    )
    return conv
# end

def conv3d_winograd_weight_transform(kernel, tile_size):
    """Weight transformation for 3D winograd

    Parameters
    ----------
    kernel: Tensor
        The raw kernel tensor with layout "NCDHW".
    tile_size: int
        Tile size of winograd transform. e.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [alpha, alpha, alpha, CO, CI]
    """
    CO, CI, KD, KH, KW = get_const_tuple(kernel.shape)

    depth_transform = 2 < KD < 8 and KD == KH

    if depth_transform:
        assert KD == KH == KW, "Only support NxNxN kernel"
    else:
        assert KH == KW, "Only supports DxNxN kernel"

    r = tile_size + KH - 1

    r_kh = te.reduce_axis((0, KH), name="r_kh")
    r_kw = te.reduce_axis((0, KW), name="r_kw")
    _, _, G = winograd_transform_matrices(tile_size, KH, kernel.dtype)
    if depth_transform:
        shape = (r, r, r, CO, CI)
        r_kd = te.reduce_axis((0, KD), name="r_kd")
        return te.compute(
            shape,
            lambda omg, eps, nu, co, ci: te.sum(
                kernel[co][ci][r_kd][r_kh][r_kw] * G[omg][r_kd] * G[eps][r_kh] * G[nu][r_kw],
                axis=[r_kd, r_kh, r_kw],
            ),
            name="transform_weight",
        )
    else:
        shape = (r, r, KD, CO, CI)
        return te.compute(
            shape,
            lambda eps, nu, d, co, ci: te.sum(
                kernel[co][ci][d][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
            ),
            name="transform_weight",
        )


@tvm.target.generic_func
def conv3d_alter_layout(attrs, inputs, tinfos, out_type):
    """Change Conv3D layout.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    out_type: type
        The output type

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level.
    """
    # not to change by default
    return None

@tvm.target.generic_func
def conv3d_infer_layout(workload, cfg):
    raise ValueError("missing register for topi.nn.conv3d_infer_layout")
