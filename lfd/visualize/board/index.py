
def run_app(directory='.', host="0.0.0.0", port=9063, debug=True):
    import os
    assert os.path.exists(directory), "The directory that was given doesn't exist"

    from dash import dcc, html
    from .app import app
    app.DIRECTORY = directory

    from .layouts import container, navbar, page
    print('layout loaded')
    from . import callbacks
    print('callbacks loaded')

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Link(rel='stylesheet', id='link'),
        navbar,
        html.Div(page, id='page-content'),
        html.Footer([container([html.P("2024")])])
    ])
    
    app.run_server(host=host, port=port, debug=debug, dev_tools_hot_reload=True)
