# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: detection.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='detection.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\x0f\x64\x65tection.proto\"\x17\n\x05\x43hunk\x12\x0e\n\x06\x62uffer\x18\x01 \x01(\x0c\"\x1a\n\x0b\x42oundingbox\x12\x0b\n\x03\x62ox\x18\x01 \x03(\x02\"D\n\x05Reply\x12\x1b\n\x05\x62oxes\x18\x01 \x03(\x0b\x32\x0c.Boundingbox\x12\x0f\n\x07\x63lasses\x18\x02 \x03(\t\x12\r\n\x05\x63onfs\x18\x03 \x03(\x02\x32)\n\x08MLServer\x12\x1d\n\x07predict\x12\x06.Chunk\x1a\x06.Reply\"\x00(\x01\x62\x06proto3')
)




_CHUNK = _descriptor.Descriptor(
  name='Chunk',
  full_name='Chunk',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='buffer', full_name='Chunk.buffer', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=42,
)


_BOUNDINGBOX = _descriptor.Descriptor(
  name='Boundingbox',
  full_name='Boundingbox',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='box', full_name='Boundingbox.box', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=70,
)


_REPLY = _descriptor.Descriptor(
  name='Reply',
  full_name='Reply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='boxes', full_name='Reply.boxes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classes', full_name='Reply.classes', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confs', full_name='Reply.confs', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=72,
  serialized_end=140,
)

_REPLY.fields_by_name['boxes'].message_type = _BOUNDINGBOX
DESCRIPTOR.message_types_by_name['Chunk'] = _CHUNK
DESCRIPTOR.message_types_by_name['Boundingbox'] = _BOUNDINGBOX
DESCRIPTOR.message_types_by_name['Reply'] = _REPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Chunk = _reflection.GeneratedProtocolMessageType('Chunk', (_message.Message,), dict(
  DESCRIPTOR = _CHUNK,
  __module__ = 'detection_pb2'
  # @@protoc_insertion_point(class_scope:Chunk)
  ))
_sym_db.RegisterMessage(Chunk)

Boundingbox = _reflection.GeneratedProtocolMessageType('Boundingbox', (_message.Message,), dict(
  DESCRIPTOR = _BOUNDINGBOX,
  __module__ = 'detection_pb2'
  # @@protoc_insertion_point(class_scope:Boundingbox)
  ))
_sym_db.RegisterMessage(Boundingbox)

Reply = _reflection.GeneratedProtocolMessageType('Reply', (_message.Message,), dict(
  DESCRIPTOR = _REPLY,
  __module__ = 'detection_pb2'
  # @@protoc_insertion_point(class_scope:Reply)
  ))
_sym_db.RegisterMessage(Reply)



_MLSERVER = _descriptor.ServiceDescriptor(
  name='MLServer',
  full_name='MLServer',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=142,
  serialized_end=183,
  methods=[
  _descriptor.MethodDescriptor(
    name='predict',
    full_name='MLServer.predict',
    index=0,
    containing_service=None,
    input_type=_CHUNK,
    output_type=_REPLY,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_MLSERVER)

DESCRIPTOR.services_by_name['MLServer'] = _MLSERVER

# @@protoc_insertion_point(module_scope)
