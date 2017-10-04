cc_binary(
    name = "main",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    deps = [
        "@opencv_local//:opencv",
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc/saved_model:loader",
    ],
)
