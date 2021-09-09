# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from .pascal_voc_pb2 import * 


class PascalVOCLabelStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CheckBlob = channel.unary_unary(
            "/object_detection.protos.PascalVOCLabel/CheckBlob",
            request_serializer= BlobFile.SerializeToString,
            response_deserializer=CheckfileRes.FromString,
        )
        self.PredictionBlob = channel.unary_unary(
            "/object_detection.protos.PascalVOCLabel/PredictionBlob",
            request_serializer=BlobReq.SerializeToString,
            response_deserializer=PascalVOCVideo.FromString,
        )
        self.PredictionFile = channel.stream_unary(
            "/object_detection.protos.PascalVOCLabel/PredictionFile",
            request_serializer=FileReq.SerializeToString,
            response_deserializer=PascalVOCVideo.FromString,
        )


class PascalVOCLabelServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CheckBlob(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PredictionBlob(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PredictionFile(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_PascalVOCLabelServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "CheckBlob": grpc.unary_unary_rpc_method_handler(
            servicer.CheckBlob,
            request_deserializer=BlobFile.FromString,
            response_serializer=CheckfileRes.SerializeToString,
        ),
        "PredictionBlob": grpc.unary_unary_rpc_method_handler(
            servicer.PredictionBlob,
            request_deserializer=BlobReq.FromString,
            response_serializer=PascalVOCVideo.SerializeToString,
        ),
        "PredictionFile": grpc.stream_unary_rpc_method_handler(
            servicer.PredictionFile,
            request_deserializer=FileReq.FromString,
            response_serializer=PascalVOCVideo.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "object_detection.protos.PascalVOCLabel", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class PascalVOCLabel(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CheckBlob(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/object_detection.protos.PascalVOCLabel/CheckBlob",
            BlobFile.SerializeToString,
            CheckfileRes.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def PredictionBlob(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/object_detection.protos.PascalVOCLabel/PredictionBlob",
            BlobReq.SerializeToString,
            PascalVOCVideo.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def PredictionFile(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_unary(
            request_iterator,
            target,
            "/object_detection.protos.PascalVOCLabel/PredictionFile",
            FileReq.SerializeToString,
            PascalVOCVideo.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
