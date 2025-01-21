from multiprocessing import freeze_support


if __name__ == "__main__":
    # this should be run before any imports to allow for
    # multiprocessing while the code is frozen (i.e. bundled in an executable)
    freeze_support()

    import tkinter as tk
    import webbrowser

    from fairsenseai import start_server

    start_server(
        make_public_url=False,
        allow_filesystem_access=False,
        prevent_thread_lock=True,
        launch_browser_on_startup=True,
    )

    # TK without a window, to get dock events on MacOS
    root = tk.Tk()
    root.title("fairsenseai")
    root.withdraw()  # remove the window

    def open_browser_window():
        webbrowser.open("http://localhost:7860")

    # Register callback for the dock icon to reopen the web app
    root.createcommand("tk::mac::ReopenApplication", open_browser_window)
    root.mainloop()
