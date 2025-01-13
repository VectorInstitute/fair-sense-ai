from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()

    from fairsenseai import start_server

    start_server(make_public_url=False)
