"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: trt_ops.cc
"""

import collections as _collections
import six as _six

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import errors as _errors
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.tf_export import kwarg_only as _kwarg_only
from tensorflow.tools.docs import doc_controls as _doc_controls


@_dispatch.add_dispatch_list
@tf_export('create_trt_resource_handle')
def create_trt_resource_handle(resource_name, name=None):
  r"""TODO: add doc.

  Args:
    resource_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "CreateTRTResourceHandle", name, _ctx.post_execution_callbacks,
        "resource_name", resource_name)
      return _result
    except _core._FallbackException:
      try:
        return create_trt_resource_handle_eager_fallback(
            resource_name=resource_name, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              create_trt_resource_handle, resource_name=resource_name,
                                          name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  resource_name = _execute.make_str(resource_name, "resource_name")
  try:
    _, _, _op = _op_def_lib._apply_op_helper(
        "CreateTRTResourceHandle", resource_name=resource_name, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          create_trt_resource_handle, resource_name=resource_name, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("resource_name", _op.get_attr("resource_name"))
  _execute.record_gradient(
      "CreateTRTResourceHandle", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def CreateTRTResourceHandle(resource_name, name=None):
  return create_trt_resource_handle(resource_name=resource_name, name=name)
CreateTRTResourceHandle.__doc__ = create_trt_resource_handle.__doc__
CreateTRTResourceHandle = _doc_controls.do_not_generate_docs(_kwarg_only(CreateTRTResourceHandle))
tf_export("raw_ops.CreateTRTResourceHandle")(CreateTRTResourceHandle)


def create_trt_resource_handle_eager_fallback(resource_name, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function create_trt_resource_handle
  """
  _ctx = ctx if ctx else _context.context()
  resource_name = _execute.make_str(resource_name, "resource_name")
  _inputs_flat = []
  _attrs = ("resource_name", resource_name)
  _result = _execute.execute(b"CreateTRTResourceHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                             name=name)
  _execute.record_gradient(
      "CreateTRTResourceHandle", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("CreateTRTResourceHandle")(None)


@_dispatch.add_dispatch_list
@tf_export('get_calibration_data_op')
def get_calibration_data_op(resource_name, name=None):
  r"""Returns calibration data for the given resource name

  Args:
    resource_name: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "GetCalibrationDataOp", name, _ctx.post_execution_callbacks,
        resource_name)
      return _result
    except _core._FallbackException:
      try:
        return get_calibration_data_op_eager_fallback(
            resource_name, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              get_calibration_data_op, resource_name=resource_name, name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op = _op_def_lib._apply_op_helper(
        "GetCalibrationDataOp", resource_name=resource_name, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          get_calibration_data_op, resource_name=resource_name, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = None
  _execute.record_gradient(
      "GetCalibrationDataOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def GetCalibrationDataOp(resource_name, name=None):
  return get_calibration_data_op(resource_name=resource_name, name=name)
GetCalibrationDataOp.__doc__ = get_calibration_data_op.__doc__
GetCalibrationDataOp = _doc_controls.do_not_generate_docs(_kwarg_only(GetCalibrationDataOp))
tf_export("raw_ops.GetCalibrationDataOp")(GetCalibrationDataOp)


def get_calibration_data_op_eager_fallback(resource_name, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function get_calibration_data_op
  """
  _ctx = ctx if ctx else _context.context()
  resource_name = _ops.convert_to_tensor(resource_name, _dtypes.string)
  _inputs_flat = [resource_name]
  _attrs = None
  _result = _execute.execute(b"GetCalibrationDataOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "GetCalibrationDataOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("GetCalibrationDataOp")(None)


@_dispatch.add_dispatch_list
@tf_export('initialize_trt_resource')
def initialize_trt_resource(resource_handle, filename, max_cached_engines_count=1, name=None):
  r"""TODO: add doc.

  Args:
    resource_handle: A `Tensor` of type `resource`.
    filename: A `Tensor` of type `string`.
    max_cached_engines_count: An optional `int`. Defaults to `1`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "InitializeTRTResource", name, _ctx.post_execution_callbacks,
        resource_handle, filename, "max_cached_engines_count",
        max_cached_engines_count)
      return _result
    except _core._FallbackException:
      try:
        return initialize_trt_resource_eager_fallback(
            resource_handle, filename,
            max_cached_engines_count=max_cached_engines_count, name=name,
            ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              initialize_trt_resource, resource_handle=resource_handle,
                                       filename=filename,
                                       max_cached_engines_count=max_cached_engines_count,
                                       name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  try:
    _, _, _op = _op_def_lib._apply_op_helper(
        "InitializeTRTResource", resource_handle=resource_handle,
                                 filename=filename,
                                 max_cached_engines_count=max_cached_engines_count,
                                 name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          initialize_trt_resource, resource_handle=resource_handle,
                                   filename=filename,
                                   max_cached_engines_count=max_cached_engines_count,
                                   name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
  _result = None
  return _result

def InitializeTRTResource(resource_handle, filename, max_cached_engines_count=1, name=None):
  return initialize_trt_resource(resource_handle=resource_handle, filename=filename, max_cached_engines_count=max_cached_engines_count, name=name)
InitializeTRTResource.__doc__ = initialize_trt_resource.__doc__
InitializeTRTResource = _doc_controls.do_not_generate_docs(_kwarg_only(InitializeTRTResource))
tf_export("raw_ops.InitializeTRTResource")(InitializeTRTResource)


def initialize_trt_resource_eager_fallback(resource_handle, filename, max_cached_engines_count=1, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function initialize_trt_resource
  """
  _ctx = ctx if ctx else _context.context()
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  resource_handle = _ops.convert_to_tensor(resource_handle, _dtypes.resource)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  _inputs_flat = [resource_handle, filename]
  _attrs = ("max_cached_engines_count", max_cached_engines_count)
  _result = _execute.execute(b"InitializeTRTResource", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("InitializeTRTResource")(None)


@_dispatch.add_dispatch_list
@tf_export('serialize_trt_resource')
def serialize_trt_resource(resource_name, filename, delete_resource=False, name=None):
  r"""TODO: add doc.

  Args:
    resource_name: A `Tensor` of type `string`.
    filename: A `Tensor` of type `string`.
    delete_resource: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "SerializeTRTResource", name, _ctx.post_execution_callbacks,
        resource_name, filename, "delete_resource", delete_resource)
      return _result
    except _core._FallbackException:
      try:
        return serialize_trt_resource_eager_fallback(
            resource_name, filename, delete_resource=delete_resource,
            name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              serialize_trt_resource, resource_name=resource_name,
                                      filename=filename,
                                      delete_resource=delete_resource,
                                      name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  if delete_resource is None:
    delete_resource = False
  delete_resource = _execute.make_bool(delete_resource, "delete_resource")
  try:
    _, _, _op = _op_def_lib._apply_op_helper(
        "SerializeTRTResource", resource_name=resource_name,
                                filename=filename,
                                delete_resource=delete_resource, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          serialize_trt_resource, resource_name=resource_name,
                                  filename=filename,
                                  delete_resource=delete_resource, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
  _result = None
  return _result

def SerializeTRTResource(resource_name, filename, delete_resource=False, name=None):
  return serialize_trt_resource(resource_name=resource_name, filename=filename, delete_resource=delete_resource, name=name)
SerializeTRTResource.__doc__ = serialize_trt_resource.__doc__
SerializeTRTResource = _doc_controls.do_not_generate_docs(_kwarg_only(SerializeTRTResource))
tf_export("raw_ops.SerializeTRTResource")(SerializeTRTResource)


def serialize_trt_resource_eager_fallback(resource_name, filename, delete_resource=False, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function serialize_trt_resource
  """
  _ctx = ctx if ctx else _context.context()
  if delete_resource is None:
    delete_resource = False
  delete_resource = _execute.make_bool(delete_resource, "delete_resource")
  resource_name = _ops.convert_to_tensor(resource_name, _dtypes.string)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  _inputs_flat = [resource_name, filename]
  _attrs = ("delete_resource", delete_resource)
  _result = _execute.execute(b"SerializeTRTResource", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("SerializeTRTResource")(None)


@_dispatch.add_dispatch_list
@tf_export('trt_engine_op')
def trt_engine_op(in_tensor, serialized_segment, OutT, workspace_size_bytes, precision_mode, segment_func="", max_cached_engines_count=1, calibration_data="", use_calibration=True, segment_funcdef_name="", cached_engine_batches=[], fixed_input_size=True, input_shapes=[], output_shapes=[], static_engine=True, name=None):
  r"""TODO: add doc.

  Args:
    in_tensor: A list of `Tensor` objects with types from: `int8`, `half`, `float32`, `int32`.
    serialized_segment: A `string`.
    OutT: A list of `tf.DTypes` from: `tf.int8, tf.half, tf.float32, tf.int32` that has length `>= 1`.
    workspace_size_bytes: An `int`.
    precision_mode: A `string` from: `"FP32", "FP16", "INT8"`.
    segment_func: An optional function decorated with @Defun. Defaults to `""`.
    max_cached_engines_count: An optional `int`. Defaults to `1`.
    calibration_data: An optional `string`. Defaults to `""`.
    use_calibration: An optional `bool`. Defaults to `True`.
    segment_funcdef_name: An optional `string`. Defaults to `""`.
    cached_engine_batches: An optional list of `ints`. Defaults to `[]`.
    fixed_input_size: An optional `bool`. Defaults to `True`.
    input_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    static_engine: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `OutT`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "TRTEngineOp", name, _ctx.post_execution_callbacks, in_tensor,
        "serialized_segment", serialized_segment, "segment_func",
        segment_func, "OutT", OutT, "max_cached_engines_count",
        max_cached_engines_count, "workspace_size_bytes",
        workspace_size_bytes, "precision_mode", precision_mode,
        "calibration_data", calibration_data, "use_calibration",
        use_calibration, "segment_funcdef_name", segment_funcdef_name,
        "cached_engine_batches", cached_engine_batches, "fixed_input_size",
        fixed_input_size, "input_shapes", input_shapes, "output_shapes",
        output_shapes, "static_engine", static_engine)
      return _result
    except _core._FallbackException:
      try:
        return trt_engine_op_eager_fallback(
            in_tensor, serialized_segment=serialized_segment,
            segment_func=segment_func, OutT=OutT,
            max_cached_engines_count=max_cached_engines_count,
            workspace_size_bytes=workspace_size_bytes,
            precision_mode=precision_mode, calibration_data=calibration_data,
            use_calibration=use_calibration,
            segment_funcdef_name=segment_funcdef_name,
            cached_engine_batches=cached_engine_batches,
            fixed_input_size=fixed_input_size, input_shapes=input_shapes,
            output_shapes=output_shapes, static_engine=static_engine,
            name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              trt_engine_op, in_tensor=in_tensor,
                             serialized_segment=serialized_segment, OutT=OutT,
                             workspace_size_bytes=workspace_size_bytes,
                             precision_mode=precision_mode,
                             segment_func=segment_func,
                             max_cached_engines_count=max_cached_engines_count,
                             calibration_data=calibration_data,
                             use_calibration=use_calibration,
                             segment_funcdef_name=segment_funcdef_name,
                             cached_engine_batches=cached_engine_batches,
                             fixed_input_size=fixed_input_size,
                             input_shapes=input_shapes,
                             output_shapes=output_shapes,
                             static_engine=static_engine, name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  serialized_segment = _execute.make_str(serialized_segment, "serialized_segment")
  if not isinstance(OutT, (list, tuple)):
    raise TypeError(
        "Expected list for 'OutT' argument to "
        "'trt_engine_op' Op, not %r." % OutT)
  OutT = [_execute.make_type(_t, "OutT") for _t in OutT]
  workspace_size_bytes = _execute.make_int(workspace_size_bytes, "workspace_size_bytes")
  precision_mode = _execute.make_str(precision_mode, "precision_mode")
  if segment_func is None:
    segment_func = ""
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  if calibration_data is None:
    calibration_data = ""
  calibration_data = _execute.make_str(calibration_data, "calibration_data")
  if use_calibration is None:
    use_calibration = True
  use_calibration = _execute.make_bool(use_calibration, "use_calibration")
  if segment_funcdef_name is None:
    segment_funcdef_name = ""
  segment_funcdef_name = _execute.make_str(segment_funcdef_name, "segment_funcdef_name")
  if cached_engine_batches is None:
    cached_engine_batches = []
  if not isinstance(cached_engine_batches, (list, tuple)):
    raise TypeError(
        "Expected list for 'cached_engine_batches' argument to "
        "'trt_engine_op' Op, not %r." % cached_engine_batches)
  cached_engine_batches = [_execute.make_int(_i, "cached_engine_batches") for _i in cached_engine_batches]
  if fixed_input_size is None:
    fixed_input_size = True
  fixed_input_size = _execute.make_bool(fixed_input_size, "fixed_input_size")
  if input_shapes is None:
    input_shapes = []
  if not isinstance(input_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_shapes' argument to "
        "'trt_engine_op' Op, not %r." % input_shapes)
  input_shapes = [_execute.make_shape(_s, "input_shapes") for _s in input_shapes]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'trt_engine_op' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if static_engine is None:
    static_engine = True
  static_engine = _execute.make_bool(static_engine, "static_engine")
  try:
    _, _, _op = _op_def_lib._apply_op_helper(
        "TRTEngineOp", in_tensor=in_tensor,
                       serialized_segment=serialized_segment, OutT=OutT,
                       workspace_size_bytes=workspace_size_bytes,
                       precision_mode=precision_mode,
                       segment_func=segment_func,
                       max_cached_engines_count=max_cached_engines_count,
                       calibration_data=calibration_data,
                       use_calibration=use_calibration,
                       segment_funcdef_name=segment_funcdef_name,
                       cached_engine_batches=cached_engine_batches,
                       fixed_input_size=fixed_input_size,
                       input_shapes=input_shapes, output_shapes=output_shapes,
                       static_engine=static_engine, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          trt_engine_op, in_tensor=in_tensor,
                         serialized_segment=serialized_segment, OutT=OutT,
                         workspace_size_bytes=workspace_size_bytes,
                         precision_mode=precision_mode,
                         segment_func=segment_func,
                         max_cached_engines_count=max_cached_engines_count,
                         calibration_data=calibration_data,
                         use_calibration=use_calibration,
                         segment_funcdef_name=segment_funcdef_name,
                         cached_engine_batches=cached_engine_batches,
                         fixed_input_size=fixed_input_size,
                         input_shapes=input_shapes,
                         output_shapes=output_shapes,
                         static_engine=static_engine, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("serialized_segment", _op.get_attr("serialized_segment"),
            "segment_func", _op.get_attr("segment_func"), "InT",
            _op.get_attr("InT"), "OutT", _op.get_attr("OutT"),
            "max_cached_engines_count",
            _op.get_attr("max_cached_engines_count"), "workspace_size_bytes",
            _op.get_attr("workspace_size_bytes"), "precision_mode",
            _op.get_attr("precision_mode"), "calibration_data",
            _op.get_attr("calibration_data"), "use_calibration",
            _op.get_attr("use_calibration"), "segment_funcdef_name",
            _op.get_attr("segment_funcdef_name"), "cached_engine_batches",
            _op.get_attr("cached_engine_batches"), "fixed_input_size",
            _op.get_attr("fixed_input_size"), "input_shapes",
            _op.get_attr("input_shapes"), "output_shapes",
            _op.get_attr("output_shapes"), "static_engine",
            _op.get_attr("static_engine"))
  _execute.record_gradient(
      "TRTEngineOp", _inputs_flat, _attrs, _result, name)
  return _result

def TRTEngineOp(in_tensor, serialized_segment, OutT, workspace_size_bytes, precision_mode, segment_func="", max_cached_engines_count=1, calibration_data="", use_calibration=True, segment_funcdef_name="", cached_engine_batches=[], fixed_input_size=True, input_shapes=[], output_shapes=[], static_engine=True, name=None):
  return trt_engine_op(in_tensor=in_tensor, serialized_segment=serialized_segment, OutT=OutT, workspace_size_bytes=workspace_size_bytes, precision_mode=precision_mode, segment_func=segment_func, max_cached_engines_count=max_cached_engines_count, calibration_data=calibration_data, use_calibration=use_calibration, segment_funcdef_name=segment_funcdef_name, cached_engine_batches=cached_engine_batches, fixed_input_size=fixed_input_size, input_shapes=input_shapes, output_shapes=output_shapes, static_engine=static_engine, name=name)
TRTEngineOp.__doc__ = trt_engine_op.__doc__
TRTEngineOp = _doc_controls.do_not_generate_docs(_kwarg_only(TRTEngineOp))
tf_export("raw_ops.TRTEngineOp")(TRTEngineOp)


def trt_engine_op_eager_fallback(in_tensor, serialized_segment, OutT, workspace_size_bytes, precision_mode, segment_func="", max_cached_engines_count=1, calibration_data="", use_calibration=True, segment_funcdef_name="", cached_engine_batches=[], fixed_input_size=True, input_shapes=[], output_shapes=[], static_engine=True, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function trt_engine_op
  """
  _ctx = ctx if ctx else _context.context()
  serialized_segment = _execute.make_str(serialized_segment, "serialized_segment")
  if not isinstance(OutT, (list, tuple)):
    raise TypeError(
        "Expected list for 'OutT' argument to "
        "'trt_engine_op' Op, not %r." % OutT)
  OutT = [_execute.make_type(_t, "OutT") for _t in OutT]
  workspace_size_bytes = _execute.make_int(workspace_size_bytes, "workspace_size_bytes")
  precision_mode = _execute.make_str(precision_mode, "precision_mode")
  if segment_func is None:
    segment_func = ""
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  if calibration_data is None:
    calibration_data = ""
  calibration_data = _execute.make_str(calibration_data, "calibration_data")
  if use_calibration is None:
    use_calibration = True
  use_calibration = _execute.make_bool(use_calibration, "use_calibration")
  if segment_funcdef_name is None:
    segment_funcdef_name = ""
  segment_funcdef_name = _execute.make_str(segment_funcdef_name, "segment_funcdef_name")
  if cached_engine_batches is None:
    cached_engine_batches = []
  if not isinstance(cached_engine_batches, (list, tuple)):
    raise TypeError(
        "Expected list for 'cached_engine_batches' argument to "
        "'trt_engine_op' Op, not %r." % cached_engine_batches)
  cached_engine_batches = [_execute.make_int(_i, "cached_engine_batches") for _i in cached_engine_batches]
  if fixed_input_size is None:
    fixed_input_size = True
  fixed_input_size = _execute.make_bool(fixed_input_size, "fixed_input_size")
  if input_shapes is None:
    input_shapes = []
  if not isinstance(input_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_shapes' argument to "
        "'trt_engine_op' Op, not %r." % input_shapes)
  input_shapes = [_execute.make_shape(_s, "input_shapes") for _s in input_shapes]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'trt_engine_op' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if static_engine is None:
    static_engine = True
  static_engine = _execute.make_bool(static_engine, "static_engine")
  _attr_InT, in_tensor = _execute.convert_to_mixed_eager_tensors(in_tensor, _ctx)
  _inputs_flat = list(in_tensor)
  _attrs = ("serialized_segment", serialized_segment, "segment_func",
  segment_func, "InT", _attr_InT, "OutT", OutT, "max_cached_engines_count",
  max_cached_engines_count, "workspace_size_bytes", workspace_size_bytes,
  "precision_mode", precision_mode, "calibration_data", calibration_data,
  "use_calibration", use_calibration, "segment_funcdef_name",
  segment_funcdef_name, "cached_engine_batches", cached_engine_batches,
  "fixed_input_size", fixed_input_size, "input_shapes", input_shapes,
  "output_shapes", output_shapes, "static_engine", static_engine)
  _result = _execute.execute(b"TRTEngineOp", len(OutT), inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TRTEngineOp", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("TRTEngineOp")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "CreateTRTResourceHandle"
#   output_arg {
#     name: "resource_handle"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "resource_name"
#     type: "string"
#   }
#   is_stateful: true
# }
# op {
#   name: "GetCalibrationDataOp"
#   input_arg {
#     name: "resource_name"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "serialized_resource"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "InitializeTRTResource"
#   input_arg {
#     name: "resource_handle"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "filename"
#     type: DT_STRING
#   }
#   attr {
#     name: "max_cached_engines_count"
#     type: "int"
#     default_value {
#       i: 1
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "SerializeTRTResource"
#   input_arg {
#     name: "resource_name"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "filename"
#     type: DT_STRING
#   }
#   attr {
#     name: "delete_resource"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "TRTEngineOp"
#   input_arg {
#     name: "in_tensor"
#     type_list_attr: "InT"
#   }
#   output_arg {
#     name: "out_tensor"
#     type_list_attr: "OutT"
#   }
#   attr {
#     name: "serialized_segment"
#     type: "string"
#   }
#   attr {
#     name: "segment_func"
#     type: "func"
#     default_value {
#       func {
#       }
#     }
#   }
#   attr {
#     name: "InT"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#     allowed_values {
#       list {
#         type: DT_INT8
#         type: DT_HALF
#         type: DT_FLOAT
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     name: "OutT"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#     allowed_values {
#       list {
#         type: DT_INT8
#         type: DT_HALF
#         type: DT_FLOAT
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     name: "max_cached_engines_count"
#     type: "int"
#     default_value {
#       i: 1
#     }
#   }
#   attr {
#     name: "workspace_size_bytes"
#     type: "int"
#   }
#   attr {
#     name: "precision_mode"
#     type: "string"
#     allowed_values {
#       list {
#         s: "FP32"
#         s: "FP16"
#         s: "INT8"
#       }
#     }
#   }
#   attr {
#     name: "calibration_data"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "use_calibration"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
#   attr {
#     name: "segment_funcdef_name"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "cached_engine_batches"
#     type: "list(int)"
#     default_value {
#       list {
#       }
#     }
#     has_minimum: true
#   }
#   attr {
#     name: "fixed_input_size"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
#   attr {
#     name: "input_shapes"
#     type: "list(shape)"
#     default_value {
#       list {
#       }
#     }
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     default_value {
#       list {
#       }
#     }
#   }
#   attr {
#     name: "static_engine"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\nJ\n\027CreateTRTResourceHandle\032\023\n\017resource_handle\030\024\"\027\n\rresource_name\022\006string\210\001\001\nE\n\024GetCalibrationDataOp\022\021\n\rresource_name\030\007\032\027\n\023serialized_resource\030\007\210\001\001\nb\n\025InitializeTRTResource\022\023\n\017resource_handle\030\024\022\014\n\010filename\030\007\"#\n\030max_cached_engines_count\022\003int\032\002\030\001\210\001\001\nW\n\024SerializeTRTResource\022\021\n\rresource_name\030\007\022\014\n\010filename\030\007\"\033\n\017delete_resource\022\004bool\032\002(\000\210\001\001\n\247\004\n\013TRTEngineOp\022\020\n\tin_tensor2\003InT\032\022\n\nout_tensor2\004OutT\"\034\n\022serialized_segment\022\006string\"\030\n\014segment_func\022\004func\032\002R\000\"\037\n\003InT\022\nlist(type)(\0010\001:\010\n\0062\004\006\023\001\003\" \n\004OutT\022\nlist(type)(\0010\001:\010\n\0062\004\006\023\001\003\"#\n\030max_cached_engines_count\022\003int\032\002\030\001\"\033\n\024workspace_size_bytes\022\003int\".\n\016precision_mode\022\006string:\024\n\022\022\004FP32\022\004FP16\022\004INT8\"\036\n\020calibration_data\022\006string\032\002\022\000\"\033\n\017use_calibration\022\004bool\032\002(\001\"\"\n\024segment_funcdef_name\022\006string\032\002\022\000\"(\n\025cached_engine_batches\022\tlist(int)\032\002\n\000(\001\"\034\n\020fixed_input_size\022\004bool\032\002(\001\"\037\n\014input_shapes\022\013list(shape)\032\002\n\000\" \n\routput_shapes\022\013list(shape)\032\002\n\000\"\031\n\rstatic_engine\022\004bool\032\002(\001")
