# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import inspect
import functools
import importlib
from operator import attrgetter
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
from tensorflow.core.framework import types_pb2

import core.operators as dlk_op
from core.data_types import DataType, Float32, Float64, Int8, Int16, Int32, \
    Int64, Uint8, Uint16, Uint32, Uint64, Bool, String
from core.exceptions import UnsupportedNode, UnsupportedDataType
from core.graph import Graph
from core.operators import Operator

DLK_DTYPE_MAP: Dict[str, Optional[DataType]] = {
    'DT_INVALID': None,
    'DT_FLOAT': Float32(),
    'DT_INT32': Int32(),
    'DT_UINT8': Uint8(),
    'DT_INT8': Int8(),
    'DT_UINT16': Uint16(),
    'DT_INT16': Int16(),
    'DT_INT64': Int64(),
    'f': Float32(),
    'i': Int32(),
    'FLOATS': None,
    'INTS': None,
    'DT_BOOL': Bool(),
    'DT_STRING': String(),
    'DT_HALF': None,
    'DT_DOUBLE': Float64(),
    'DT_UINT32': Uint32(),
    'DT_UINT64': Uint64(),
    'DT_COMPLEX64': None,
    'DT_COMPLEX128': None,
    's': String(),
    'STRINGS': None,
    'TENSOR': None,
    'GRAPH': None,
    't': None,
    'g': None,
    'TENSORS': None,
    'GRAPHS': None,
}

DLK_OPERATOR_MAP: Dict[str, str] = {
    'Conv2D': 'Conv',
    'FusedBatchNorm': 'BatchNormalization',
    'AvgPool': 'AveragePool',
    'BiasAdd': 'Add',
    'ConcatV2': 'ConcatOnDepth'
}

TF_ATTR_TYPE_MAP: Dict[str, str] = {
    'padding': 's',
    'data_format': 's',
    '_gradient_op_type': 's',
    'strides': 'list.i',
    'ksize': 'list.i',
    'epsilon': 'f',
    'is_training': 'b',
    'use_cudnn_on_gpu': 'b',
    'transpose_b': 'b',
    'transpose_a': 'b',
    'block_size': 'i',
    'num_split': 'i',
    'N': 'i',
    '_output_shapes': 'list.shape',
    'T': 'type',
    'Tshape': 'type',
    'Tpaddings': 'type',
    'Tidx': 'type',
    'dtype': 'type',
    'shape': 'shape.dim',
    'value': 'tensor',
    '_class': 'list.s',
}


class TFProtoWrapper(object):
    DATA_TYPE_MAP = {v: k for k, v in types_pb2.DataType.items()}

    def __init__(self, tf_node) -> None:
        self.tf_nd = tf_node
        self.attributes: List = []
        self.attrs_to_value: Dict[str, Any] = {}
        for key in self.tf_nd.attr.keys():
            self.attributes.append(key)
        self.get_attribute_values()

    @property
    def name(self) -> str:
        """Return the name of the node."""
        return self.tf_nd.name.replace('/', '_').replace('-', '_')

    @property
    def op_type(self) -> str:
        """Return the op type of the node."""
        return self.tf_nd.op

    @property
    def nodedef_object(self):
        """Return node def object."""
        return self.tf_nd

    @property
    def is_placeholder(self) -> bool:
        """Check op is place holder or not."""
        return self.tf_nd.op == 'Placeholder'

    @property
    def tensor_type(self) -> int:
        """Get tensor type info."""
        if self.tf_nd.op in ['QTZ_binary_mean_scaling',
                             'QTZ_linear_mid_tread_half',
                             'QTZ_binary_channel_wise_mean_scaling']:
            typep = 1
        else:
            typep = self.attrs_to_value['T']
        return typep

    @property
    def inputs(self) -> List[str]:
        """Return the name of corresponding inputs to the node."""
        return [x.replace('/', '_').replace('-', '_').split(':', 1)[0] for x in self.tf_nd.input]

    def get_shape(self) -> List[str]:
        return self.attrs_to_value['_output_shapes']

    def get_dtype(self) -> DataType:
        """Get data type info."""
        dtype_str = self.DATA_TYPE_MAP[self.tensor_type]
        if DLK_DTYPE_MAP[dtype_str] is None:
            raise UnsupportedDataType(f'Type {dtype_str} is not supported.')
        return DLK_DTYPE_MAP[dtype_str]

    def get_attribute_values(self) -> None:
        """Return the attributes data corresponding to the node."""
        for attr in self.attributes:
            x = attrgetter(TF_ATTR_TYPE_MAP[attr])
            value = attrgetter(TF_ATTR_TYPE_MAP[attr])(self.tf_nd.attr[attr])
            if attr == '_output_shapes':
                out_shapes = []
                if value:
                    for d in value:
                        shape = []
                        for v in range(len(d.dim)):
                            shape.append(d.dim[v].size)
                        out_shapes.append(shape)
                else:
                    raise ValueError(f'{self.name} does not have output shapes.')

                if len(out_shapes) > 1 and self.tf_nd.op == 'Split' and not out_shapes[1:] == out_shapes[:-1]:
                        raise ValueError(f'{self.name} does not have identical output(s) shape.')
                self.attrs_to_value[attr] = out_shapes[0]
            elif attr in ['strides', 'ksize']:
                self.attrs_to_value[attr] = value[1:3]
            elif attr in ['padding', 'data_format']:
                self.attrs_to_value[attr] = value.decode(encoding='utf-8')
            elif attr == 'shape':
                self.attrs_to_value[attr] = [d.size for d in value]
            else:
                self.attrs_to_value[attr] = value


class Node(TFProtoWrapper):

    def __init__(self, op_nd) -> None:
        super().__init__(op_nd)
        self.nd_ = op_nd

    def get_format(self) -> Optional[str]:
        """Get the output data format info."""
        return self.attrs_to_value['data_format'] if 'data_format' in self.attrs_to_value.keys() else None


class Input(TFProtoWrapper):
    # Data conversion table from binary to numpy
    _TF_TO_NP = {
        types_pb2.DT_HALF:
            np.float16,
        types_pb2.DT_FLOAT:
            np.float32,
        types_pb2.DT_DOUBLE:
            np.float64,
        types_pb2.DT_INT32:
            np.int32,
        types_pb2.DT_UINT16:
            np.uint16,
        types_pb2.DT_INT16:
            np.int16,
        types_pb2.DT_INT8:
            np.int8,
        types_pb2.DT_STRING:
            np.object,
        types_pb2.DT_COMPLEX64:
            np.complex64,
        types_pb2.DT_COMPLEX128:
            np.complex128,
        types_pb2.DT_INT64:
            np.int64,
        types_pb2.DT_BOOL:
            np.bool,
    }

    def __init__(self, in_nd) -> None:
        super().__init__(in_nd)
        if not self.is_placeholder:
            self.tensor = self.attrs_to_value['value']

    def get_dtype(self) -> DataType:
        """Get data type info."""
        if self.is_placeholder:
            dtype_str = type(self).DATA_TYPE_MAP[self.attrs_to_value['dtype']]
        else:
            dtype_str = type(self).DATA_TYPE_MAP[self.tensor.dtype]

        if DLK_DTYPE_MAP[dtype_str] is None:
            raise UnsupportedDataType(f'Type {dtype_str} is not supported.')

        return DLK_DTYPE_MAP[dtype_str]

    def get_data(self) -> np.ndarray:
        """Get data in numpy format."""
        if self.is_placeholder:
            raise ValueError(
                f'{self.name} is a placeholder, which does\'t have data...')
        dtype = type(self)._TF_TO_NP[self.tensor.dtype]
        if self.tensor.tensor_content:
            return np.frombuffer(self.tensor.tensor_content, dtype=dtype).copy().reshape(self.get_shape())
        else:
            if self.tensor.dtype == 3:
                return np.asarray(self.tensor.int_val, dtype=dtype)
            if self.tensor.dtype == 1:
                return np.asarray(self.tensor.float_val, dtype=dtype)

    def get_shape(self) -> List[int]:
        """Get shape info."""
        shape_sets: List = None
        if len(self.attrs_to_value.keys() & {'_output_shapes', 'shape'}) > 0:
            for x in self.attrs_to_value.keys() & {'_output_shapes', 'shape'}:
                if self.attrs_to_value[x]:
                    shape_sets = self.attrs_to_value[x]
        if shape_sets is not None:
            return shape_sets
        elif self.attrs_to_value['value'].tensor_shape.dim:
            return [d.size for d in self.attrs_to_value['value'].tensor_shape.dim]
        else:
            return [self.get_data().size]


class Output(TFProtoWrapper):
    def __init__(self, out_nd) -> None:
        super().__init__(out_nd)


class Importer(object):

    @classmethod
    def make_graph(cls, tf_mp) -> Graph:
        importer = Importer(tf_mp)
        graph = Graph()

        importer.add_all_nodes(graph)
        return graph

    @staticmethod
    def convert_operator(op_type: str) -> str:
        """Convert Tensorflow operator type to DLK's one."""
        dlk_op_type = DLK_OPERATOR_MAP.get(op_type)
        return dlk_op_type if dlk_op_type else op_type

    def __init__(self, tf_mp) -> None:
        """Init the graph.
        Prameters
        ---------
        tf_mp : GraphDef
            GraphDef object
        """

        self.tf_gp = tf_mp
        self.in_lst: List[Input] = []
        self.out_lst: List[Output] = []
        self.node_lst: List[Node] = []
        self.node_dic: Dict[str, Any] = {}
        node_obj: Any = None
        for node in self.tf_gp.node:
            if node.op == 'Const' or node.op == 'Placeholder':
                node_obj = Input(node)
                self.in_lst.append(node_obj)
            else:
                if node.name == "output":
                    node_obj = Output(node)
                    self.out_lst.append(node_obj)
                else:
                    node_obj = Node(node)
                    self.node_lst.append(node_obj)
            self.node_dic[node_obj.name] = node_obj

        self.validate_tf()

        self.input_hash_table = self.construct_input_hash_table(
            self.node_lst, self.out_lst, self.in_lst)

    def validate_tf(self) -> None:
        """Validate if the GraphDef object is proper."""
        gp = self.tf_gp
        assert len(gp.node) == (len(self.in_lst) + len(self.node_lst) + len(self.out_lst))

    def create_new_op(self, node: Any, op_dic: Dict[str, Operator], current_format: str,
                      input_format_list: List[str], nodes_to_remove) -> Operator:
        """Create new operators with Node, Input(Constant), Output."""
        new_op: Operator

        if isinstance(node, Node):  # operator nodes
            new_op = self.create_new_node(node, op_dic, current_format, input_format_list, nodes_to_remove)

        else:  # Input, Output or Constant
            shape: List[int] = list(map(int, node.get_shape()))
            dtype = node.get_dtype()

            # print(node.op_type, ' name:', node.name, ' shape:', shape)

            if isinstance(node, Input):
                if node.is_placeholder:  # op_type = 'Input'
                    shape = list(map(int, node.get_shape()))
                    new_op = dlk_op.Input(
                        node.name,
                        shape,
                        dtype,
                        dimension_format=current_format
                    )

                else:  # op_type = 'Constant'
                    data = node.get_data()
                    new_op = dlk_op.Constant(
                        node.name,
                        dtype,
                        data,
                        dimension_format=current_format
                    )

            elif isinstance(node, Output):  # op_type = 'Output'
                # get input operator
                input_ops = {k: op_dic[n.name] for n, k in zip(
                    self.find_inputs(node), dlk_op.Output.input_names)}

                new_op = dlk_op.Output(
                    node.name,
                    shape,
                    dtype,
                    input_ops,
                )

        return new_op

    def add_all_nodes(self, graph: Graph) -> None:
        visited: Set[Any] = set()
        added: Dict[str, Operator] = {}
        nodes_to_remove = []
        self.add_node_to_graph_recursive(self.out_lst[0], graph, visited, added, 'NHWC', nodes_to_remove)
        for node in nodes_to_remove:
            graph.remove_op(node)

    def _get_format(self, node: Any, output_format: str) -> Tuple[str, List[str]]:
        """Get the dimension format, like 'NCHW', 'HWCN', 'NHWC', etc for operators.
        Always use the format from tensorflow, if the layout format is not defined,
        then propagate the format from the output. Special case such as:
        - 'Conv': by default of tensorflow, input is 'NHWC', and kernel 'HWCN'
        https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        - 'QTZ_binary_mean_scaling', 'QTZ_binary_channel_wise_mean_scaling':
        kernel quantizer is also in HWCN
        - 'Transpose': depending on the perm attribute
        """

        _default_format = 'NHWC'
        _default_w_format = 'HWCN'

        rank_to_format = {1: 'C', 2: 'WC', 3: 'HWC', 4: 'NHWC'}

        def guess_node_format(input_node: Any) -> str:
            """Provide the node format from references
            By means of guessing, the rank decides the input layout format of current node.
            For instance, if the input has same rank as the output of the current node,
            then the input is assumed to have same layout format as the output, otherwise,
            the format follows 'C', 'WC', and 'HWC' respectively of rank 1, 2, 3.
            """
            node_rank = len(input_node.get_shape())
            return out_format if node_rank == len(out_format) else rank_to_format[node_rank]

        if isinstance(node, Node):
            if node.get_format() is None:
                out_format = output_format or _default_format
            else:
                out_format = node.get_format()

            op_type = self.convert_operator(node.op_type)
            if op_type == 'Conv':
                return out_format, [out_format, _default_w_format, 'C']
            elif op_type in ['QTZ_binary_mean_scaling', 'QTZ_binary_channel_wise_mean_scaling']:
                return _default_w_format, [_default_w_format]
            elif op_type == 'Transpose':
                perm = list(node.attrs_to_value['perm'])
                inv_perm = [perm.index(i) for i in range(len(perm))]  # inverse permutation
                input_format = functools.reduce(
                    lambda x, y: x + y, [output_format[i] for i in inv_perm])
                return output_format, [input_format]
            else:
                input_format_list = []
                for node_name in node.inputs:
                    node_object = self.node_dic[node_name]
                    if not isinstance(node_object, Input):
                        node_object_format = node_object.get_format() if node_object.get_format() is not None else \
                            guess_node_format(node_object)
                    else:
                        node_object_format = guess_node_format(node_object)
                    input_format_list.append(node_object_format)
                return out_format, input_format_list
        else:
            return output_format, [output_format]

    def add_node_to_graph_recursive(self, current: Any, graph: Graph, visited: Set[Any], added: Dict[str, Operator],
                                    data_format: str, nodes_to_remove) \
            -> Operator:
        if current in visited:
            return added[current.name]
            # return current

        added_op_dic: Dict[str, Operator] = {}

        current_format, input_formats = self._get_format(current, data_format)
        inputs = self.find_inputs(current)
        for in_put, in_format in zip(inputs, input_formats):
            in_op = self.add_node_to_graph_recursive(in_put, graph, visited, added, in_format, nodes_to_remove)
            added_op_dic[in_op.name] = in_op

        op = self.create_new_op(current, added_op_dic, current_format, input_formats, nodes_to_remove)

        graph.add_op(op)

        visited.add(current)
        added[op.name] = op
        return op

    def construct_input_hash_table(self, node_list: List[Any], out_list: List[Any], input_list: List[Any]) \
            -> Dict[str, Any]:
        hash_table: Dict[str, Any] = defaultdict(lambda: [])

        for x in input_list:
            hash_table[x.name].append(x)

        for node in node_list + out_list:
            for idx in node.inputs:
                hash_table[node.name].append(self.node_dic[idx])

        return hash_table

    def find_inputs(self, node: Any) -> List[Any]:
        inputs: List[Any] = []
        if not isinstance(node, Input):
            for idx in node.inputs:
                inputs.append(self.node_dic[idx])

        return inputs

    def create_new_node(self, node: Node, op_dic: Dict[str, Operator], current_format: str,
                        input_format_list: List[str], nodes_to_remove: List[Operator]) -> Operator:
        """Create a new operator node. This might be tooooo long code...
        Parameters
        ----------
        node : Node
            TF node corresponding to the operator
        op_dic : Dict from str to Operator
            Dict of preceding operators
        current_format : Dict from str to str
            Dict of data format of current node
        input_format_list : Dict from str to str
            Dict of data format of corresponding inputs of current node
        nodes_to_remove : List of operators
            Dict of data format of corresponding inputs of current node
        Returns
        -------
        new_op : Operator
            Newly created dlk operator
        """
        op_type = self.convert_operator(node.op_type)
        try:
            module = importlib.import_module('core.operators')
            dlk_operator = getattr(module, op_type)
        except AttributeError:
            message = f'Operator {op_type} is not supported.'
            raise UnsupportedNode(message)

        def get_inputs(cdef: Type[Operator], current_node: Any) -> Dict[str, Operator]:
            input_names = cdef.input_names
            in_ops: Dict[str, Operator] = {}
            for n, op in zip(input_names, current_node.inputs):
                in_ops[n] = op_dic[op]
            return in_ops

        input_ops = get_inputs(dlk_operator, node)

        def infer_shape(attrs: Dict[str, Any]) -> List[int]:
            shape_dict = {n: input_ops[n].shape for n in dlk_operator.input_names if input_ops.get(n)}
            return dlk_operator.infer_shape(shape_dict, current_format, input_format_list, attrs)

        def infer_dtype() -> DataType:
            if node.get_dtype() is not None:
                return node.get_dtype()  # type: ignore
            else:
                return list(input_ops.values())[0].dtype

        shape: List[int] = list(map(int, node.get_shape()))
        if not shape:
            shape = infer_shape(node.attrs_to_value)
        dtype = infer_dtype()

        # dlk_operator = getattr(sys.modules[__name__], op_type)
        members = inspect.signature(dlk_operator)
        basic_args: Dict[str, Any] = {'name': node.name,
                                      'shape': shape,
                                      'dtype': dtype,
                                      'input_ops': input_ops,
                                      'dimension_format': current_format}
        params: Dict[str, Any] = {}
        for args in members.parameters.keys():
            if args in basic_args.keys():
                params[args] = basic_args[args]
            if args in node.attrs_to_value.keys():
                params[args] = node.attrs_to_value[args]
            elif args in ['pads', 'kernel_shape']:
                strides = node.attrs_to_value['strides']
                padding = node.attrs_to_value['padding']

                if 'ksize' in node.attrs_to_value.keys():
                    ksize = node.attrs_to_value['ksize']
                    filt_h = ksize[0]
                    filt_w = ksize[1]
                else:
                    filt_h = input_ops['W'].shape[input_format_list[1].index('H')]
                    filt_w = input_ops['W'].shape[input_format_list[1].index('W')]

                in_h = input_ops['X'].height
                in_w = input_ops['X'].width
                stride_h = strides[0]
                stride_w = strides[1]

                if padding == 'SAME':
                    if in_h % stride_h == 0:
                        pad_along_height = max(filt_h - stride_h, 0)
                    else:
                        pad_along_height = max(filt_h - (in_h % stride_h), 0)
                    if in_w % stride_w == 0:
                        pad_along_width = max(filt_w - stride_w, 0)
                    else:
                        pad_along_width = max(filt_w - (in_w % stride_w), 0)

                    pad_top = pad_along_height // 2
                    pad_bottom = pad_along_height - pad_top
                    pad_left = pad_along_width // 2
                    pad_right = pad_along_width - pad_left

                    pads: List[int] = [pad_top, pad_bottom, pad_left, pad_right]

                elif padding == 'VALID':
                    pads: List[int] = [0, 0, 0, 0]

                else:
                    raise ValueError('f{op_type} {node.name} doesn\'t have the supported padding.')

                params['pads'] = pads
                params['kernel_shape'] = [filt_h, filt_w]

        new_op: Operator = dlk_operator(**params)

        return new_op
