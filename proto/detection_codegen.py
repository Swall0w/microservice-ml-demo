from grpc.tools import protoc


protoc.main(
    (
        '',
        '-I.',
        '--python_out=../lib/',
        '--grpc_python_out=../lib/',
        './detection.proto',
    )
)
