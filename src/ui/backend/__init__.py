def run(port: int = 5000, debug=False):
    from . import system_info
    from . import routes
    # from . import watcher

    try:
        system_info.run(port + 1)
        # watcher.run()
        routes.run(port=port, debug=debug)
    except KeyboardInterrupt:
        print("Shutting down server...")
        exit(0)
