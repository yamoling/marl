def run(port: int = 5000, debug=False):
    from . import system_info
    from . import routes

    try:
        system_info.run(port + 1)
        routes.run(port=port, debug=debug)
    except KeyboardInterrupt:
        print("Shutting down server...")
        exit(0)
