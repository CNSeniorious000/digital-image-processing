from typer import Typer

dl_app = Typer(name="dl", help="Deep Learning Tasks", no_args_is_help=True)


@dl_app.command()
def task1():
    from dl.task1 import main

    main()


@dl_app.command()
def task2():
    pass


@dl_app.command()
def task3():
    pass
