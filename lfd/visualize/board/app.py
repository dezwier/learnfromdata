from dash import Dash
#import dash_auth
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    title = 'Advanced Analytics',
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True
)

server = app.server
#credentials = pd.read_csv("credentials.csv").to_dict('records')
#VALID_USERNAME_PASSWORD_PAIRS = {i['username']:i['password'] for i in credentials}
#VALID_USERNAME_PASSWORD_PAIRS = {'user':'dnabelins'}
#auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
