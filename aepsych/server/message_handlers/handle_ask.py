import logging
import aepsych.utils_logging as utils_logging

# from aepsych.server.server import AEPsychServer


logger = utils_logging.getLogger(logging.INFO)


def handle_ask_v01(server, request):
    """Returns dictionary with two entries:
    "config" -- dictionary with config (keys are strings, values are floats)
    "is_finished" -- bool, true if the strat is finished
    """
    logger.debug("got ask message!")
    if server._pregen_asks:
        params = server._pregen_asks.pop()
    else:
        params = ask(server)

    new_config = {"config": params, "is_finished": server.strat.finished}
    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="ask", request=request
        )
    return new_config


def handle_ask(server, request):
    logger.debug("got ask message!")

    if server._pregen_asks:
        return server._pregen_asks.pop()

    new_config = ask(server)

    if not server.is_performing_replay:
        server.db.record_message(
            master_table=server._db_master_record, type="ask", request=request
        )

    return new_config


def ask(server):
    """get the next point to query from the model
    Returns:
        dict -- new config dict (keys are strings, values are floats)
    """
    if server.skip_computations:
        # HACK to makke sure strategies finish correctly
        server.strat._strat._count += 1
        if server.strat._strat.finished:
            server.strat._make_next_strat()
        return None

    if not server.use_ax:
        # index by [0] is temporary HACK while serverside
        # doesn't handle batched ask
        next_x = server.strat.gen()[0]
        return server._tensor_to_config(next_x)

    next_x = server.strat.gen()
    return next_x
