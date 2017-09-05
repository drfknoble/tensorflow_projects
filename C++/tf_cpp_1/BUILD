cc_binary(
    name = "main",
    srcs = glob([
        "*.c*",
        "*.h",
    ]),
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/core:tensorflow",
    ],
    data = [
        "//tensorflow/tf_cpp_1:test_image.png",
    ],
)
