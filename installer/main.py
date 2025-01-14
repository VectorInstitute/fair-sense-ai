from multiprocessing import freeze_support


if __name__ == "__main__":
    # this should be run before any imports to allow for
    # multiprocessing while the code is frozen (i.e. bundled in an executable)
    freeze_support()

    import time
    import tkinter as tk
    import webbrowser
    from threading import Thread

    from fairsenseai import start_server

    def open_browser_window():
        webbrowser.open("http://localhost:7860")

    # Starting in a daemon thread so the main thread is taken by TK
    server_thread = Thread(target=start_server, args=(False, False,), daemon=True)
    server_thread.start()

    # TK without a window, to get dock events on MacOS
    root = tk.Tk()
    root.title("fairsenseai")
    root.withdraw()  # remove the window

    # HACK: Sleep for 30s to allow the server to start up
    sleep_time = 30
    print(f"Sleeping for {sleep_time}s...")
    time.sleep(sleep_time)
    # Then open the browser window
    open_browser_window()

    # Register callback for the dock icon to reopen the web app
    root.createcommand("tk::mac::ReopenApplication", open_browser_window)
    root.mainloop()
