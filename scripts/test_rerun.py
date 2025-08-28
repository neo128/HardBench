import rerun as rr

# 初始化（必须在启动服务前调用）
rr.init("your_app_name", spawn=False)
mode = "distant"
if mode == "distant":
    # 启动 gRPC 服务
    grpc_uri = rr.serve_grpc(
        grpc_port=9876,
        server_memory_limit="25%"  # 或 "0B"（同一机器时推荐）
    )
    print(f"gRPC 服务已启动，URI: {grpc_uri}")
    
    # 启动 Web 前端服务（使用 web_port 参数）
    rr.serve_web_viewer(
        open_browser=True,
        web_port=9090,  # 将 port 改为 web_port
        connect_to=grpc_uri  # 显式指定连接到 gRPC 服务（可选，但推荐）
    )


