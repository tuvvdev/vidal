# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pascal_voc.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='pascal_voc.proto',
  package='object_detection.protos',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10pascal_voc.proto\x12\x17object_detection.protos\"a\n\x0b\x42oundingBox\x12\x0b\n\x03\x63ls\x18\x01 \x01(\x05\x12\x0c\n\x04xmin\x18\x02 \x01(\x05\x12\x0c\n\x04ymin\x18\x03 \x01(\x05\x12\x0c\n\x04xmax\x18\x04 \x01(\x05\x12\x0c\n\x04ymax\x18\x05 \x01(\x05\x12\r\n\x05score\x18\x06 \x01(\x02\"S\n\x0ePascalVOCImage\x12\r\n\x05index\x18\x01 \x01(\x05\x12\x32\n\x04\x62\x62ox\x18\x02 \x03(\x0b\x32$.object_detection.protos.BoundingBox\"\x89\x01\n\x0ePascalVOCVideo\x12\x0e\n\x06object\x18\x01 \x01(\t\x12\x10\n\x08\x66ilename\x18\x02 \x01(\t\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\x12\x36\n\x05image\x18\x05 \x03(\x0b\x32\'.object_detection.protos.PascalVOCImage\"\x1c\n\x08\x46ilename\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\"W\n\x07\x46ileReq\x12\x10\n\x06\x62uffer\x18\x01 \x01(\x0cH\x00\x12\x33\n\x04info\x18\x02 \x01(\x0b\x32#.object_detection.protos.DetectFileH\x00\x42\x05\n\x03one\"-\n\x08\x42lobFile\x12\x0f\n\x07project\x18\x01 \x01(\t\x12\x10\n\x08\x66ilename\x18\x02 \x01(\t\"T\n\x07\x42lobReq\x12\x0f\n\x07project\x18\x01 \x01(\t\x12\x38\n\x0b\x64\x65tect_file\x18\x02 \x01(\x0b\x32#.object_detection.protos.DetectFile\">\n\nDetectFile\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\x12\x0e\n\x06object\x18\x02 \x01(\t\x12\x0e\n\x06stride\x18\x03 \x01(\x05\"\x1b\n\x0c\x43heckfileRes\x12\x0b\n\x03res\x18\x01 \x01(\x08\x32\xa9\x02\n\x0ePascalVOCLabel\x12W\n\tCheckBlob\x12!.object_detection.protos.BlobFile\x1a%.object_detection.protos.CheckfileRes\"\x00\x12]\n\x0ePredictionBlob\x12 .object_detection.protos.BlobReq\x1a\'.object_detection.protos.PascalVOCVideo\"\x00\x12_\n\x0ePredictionFile\x12 .object_detection.protos.FileReq\x1a\'.object_detection.protos.PascalVOCVideo\"\x00(\x01\x62\x06proto3'
)




_BOUNDINGBOX = _descriptor.Descriptor(
  name='BoundingBox',
  full_name='object_detection.protos.BoundingBox',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='cls', full_name='object_detection.protos.BoundingBox.cls', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='xmin', full_name='object_detection.protos.BoundingBox.xmin', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ymin', full_name='object_detection.protos.BoundingBox.ymin', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='xmax', full_name='object_detection.protos.BoundingBox.xmax', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ymax', full_name='object_detection.protos.BoundingBox.ymax', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='score', full_name='object_detection.protos.BoundingBox.score', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=142,
)


_PASCALVOCIMAGE = _descriptor.Descriptor(
  name='PascalVOCImage',
  full_name='object_detection.protos.PascalVOCImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='index', full_name='object_detection.protos.PascalVOCImage.index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bbox', full_name='object_detection.protos.PascalVOCImage.bbox', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=144,
  serialized_end=227,
)


_PASCALVOCVIDEO = _descriptor.Descriptor(
  name='PascalVOCVideo',
  full_name='object_detection.protos.PascalVOCVideo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='object', full_name='object_detection.protos.PascalVOCVideo.object', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='filename', full_name='object_detection.protos.PascalVOCVideo.filename', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='object_detection.protos.PascalVOCVideo.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='object_detection.protos.PascalVOCVideo.height', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image', full_name='object_detection.protos.PascalVOCVideo.image', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=230,
  serialized_end=367,
)


_FILENAME = _descriptor.Descriptor(
  name='Filename',
  full_name='object_detection.protos.Filename',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename', full_name='object_detection.protos.Filename.filename', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=369,
  serialized_end=397,
)


_FILEREQ = _descriptor.Descriptor(
  name='FileReq',
  full_name='object_detection.protos.FileReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='buffer', full_name='object_detection.protos.FileReq.buffer', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='info', full_name='object_detection.protos.FileReq.info', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='one', full_name='object_detection.protos.FileReq.one',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=399,
  serialized_end=486,
)


_BLOBFILE = _descriptor.Descriptor(
  name='BlobFile',
  full_name='object_detection.protos.BlobFile',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='project', full_name='object_detection.protos.BlobFile.project', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='filename', full_name='object_detection.protos.BlobFile.filename', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=488,
  serialized_end=533,
)


_BLOBREQ = _descriptor.Descriptor(
  name='BlobReq',
  full_name='object_detection.protos.BlobReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='project', full_name='object_detection.protos.BlobReq.project', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='detect_file', full_name='object_detection.protos.BlobReq.detect_file', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=535,
  serialized_end=619,
)


_DETECTFILE = _descriptor.Descriptor(
  name='DetectFile',
  full_name='object_detection.protos.DetectFile',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename', full_name='object_detection.protos.DetectFile.filename', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='object', full_name='object_detection.protos.DetectFile.object', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='stride', full_name='object_detection.protos.DetectFile.stride', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=621,
  serialized_end=683,
)


_CHECKFILERES = _descriptor.Descriptor(
  name='CheckfileRes',
  full_name='object_detection.protos.CheckfileRes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='res', full_name='object_detection.protos.CheckfileRes.res', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=685,
  serialized_end=712,
)

_PASCALVOCIMAGE.fields_by_name['bbox'].message_type = _BOUNDINGBOX
_PASCALVOCVIDEO.fields_by_name['image'].message_type = _PASCALVOCIMAGE
_FILEREQ.fields_by_name['info'].message_type = _DETECTFILE
_FILEREQ.oneofs_by_name['one'].fields.append(
  _FILEREQ.fields_by_name['buffer'])
_FILEREQ.fields_by_name['buffer'].containing_oneof = _FILEREQ.oneofs_by_name['one']
_FILEREQ.oneofs_by_name['one'].fields.append(
  _FILEREQ.fields_by_name['info'])
_FILEREQ.fields_by_name['info'].containing_oneof = _FILEREQ.oneofs_by_name['one']
_BLOBREQ.fields_by_name['detect_file'].message_type = _DETECTFILE
DESCRIPTOR.message_types_by_name['BoundingBox'] = _BOUNDINGBOX
DESCRIPTOR.message_types_by_name['PascalVOCImage'] = _PASCALVOCIMAGE
DESCRIPTOR.message_types_by_name['PascalVOCVideo'] = _PASCALVOCVIDEO
DESCRIPTOR.message_types_by_name['Filename'] = _FILENAME
DESCRIPTOR.message_types_by_name['FileReq'] = _FILEREQ
DESCRIPTOR.message_types_by_name['BlobFile'] = _BLOBFILE
DESCRIPTOR.message_types_by_name['BlobReq'] = _BLOBREQ
DESCRIPTOR.message_types_by_name['DetectFile'] = _DETECTFILE
DESCRIPTOR.message_types_by_name['CheckfileRes'] = _CHECKFILERES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BoundingBox = _reflection.GeneratedProtocolMessageType('BoundingBox', (_message.Message,), {
  'DESCRIPTOR' : _BOUNDINGBOX,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.BoundingBox)
  })
_sym_db.RegisterMessage(BoundingBox)

PascalVOCImage = _reflection.GeneratedProtocolMessageType('PascalVOCImage', (_message.Message,), {
  'DESCRIPTOR' : _PASCALVOCIMAGE,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.PascalVOCImage)
  })
_sym_db.RegisterMessage(PascalVOCImage)

PascalVOCVideo = _reflection.GeneratedProtocolMessageType('PascalVOCVideo', (_message.Message,), {
  'DESCRIPTOR' : _PASCALVOCVIDEO,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.PascalVOCVideo)
  })
_sym_db.RegisterMessage(PascalVOCVideo)

Filename = _reflection.GeneratedProtocolMessageType('Filename', (_message.Message,), {
  'DESCRIPTOR' : _FILENAME,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.Filename)
  })
_sym_db.RegisterMessage(Filename)

FileReq = _reflection.GeneratedProtocolMessageType('FileReq', (_message.Message,), {
  'DESCRIPTOR' : _FILEREQ,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.FileReq)
  })
_sym_db.RegisterMessage(FileReq)

BlobFile = _reflection.GeneratedProtocolMessageType('BlobFile', (_message.Message,), {
  'DESCRIPTOR' : _BLOBFILE,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.BlobFile)
  })
_sym_db.RegisterMessage(BlobFile)

BlobReq = _reflection.GeneratedProtocolMessageType('BlobReq', (_message.Message,), {
  'DESCRIPTOR' : _BLOBREQ,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.BlobReq)
  })
_sym_db.RegisterMessage(BlobReq)

DetectFile = _reflection.GeneratedProtocolMessageType('DetectFile', (_message.Message,), {
  'DESCRIPTOR' : _DETECTFILE,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.DetectFile)
  })
_sym_db.RegisterMessage(DetectFile)

CheckfileRes = _reflection.GeneratedProtocolMessageType('CheckfileRes', (_message.Message,), {
  'DESCRIPTOR' : _CHECKFILERES,
  '__module__' : 'pascal_voc_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.CheckfileRes)
  })
_sym_db.RegisterMessage(CheckfileRes)



_PASCALVOCLABEL = _descriptor.ServiceDescriptor(
  name='PascalVOCLabel',
  full_name='object_detection.protos.PascalVOCLabel',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=715,
  serialized_end=1012,
  methods=[
  _descriptor.MethodDescriptor(
    name='CheckBlob',
    full_name='object_detection.protos.PascalVOCLabel.CheckBlob',
    index=0,
    containing_service=None,
    input_type=_BLOBFILE,
    output_type=_CHECKFILERES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PredictionBlob',
    full_name='object_detection.protos.PascalVOCLabel.PredictionBlob',
    index=1,
    containing_service=None,
    input_type=_BLOBREQ,
    output_type=_PASCALVOCVIDEO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PredictionFile',
    full_name='object_detection.protos.PascalVOCLabel.PredictionFile',
    index=2,
    containing_service=None,
    input_type=_FILEREQ,
    output_type=_PASCALVOCVIDEO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_PASCALVOCLABEL)

DESCRIPTOR.services_by_name['PascalVOCLabel'] = _PASCALVOCLABEL

# @@protoc_insertion_point(module_scope)
