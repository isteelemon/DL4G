# HSLU
#
# Created by Thomas Koller on 7/30/2020
#
"""
Example how to use flask to create a service for one or more players
"""
import logging

from jass.agents.Agent_rulebased import Agent_rulebased
from jass.agents.MCTS_agent import MCTS_agent
from jass.service.player_service_app import PlayerServiceApp
from jass.agents.agent_random_schieber import AgentRandomSchieber


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('random1', AgentRandomSchieber())
    app.add_player('random2', AgentRandomSchieber())
    app.add_player('random3', AgentRandomSchieber())
    app.add_player('JASSMASTER3000', MCTS_agent())

    return app


if __name__ == '__main__':
   app = create_app()
   app.run("0.0.0.0", 8080)

