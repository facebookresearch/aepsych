import logging
from aepsych.version import __version__
from aepsych.config import Config
import aepsych.utils_logging as utils_logging
from aepsych.strategy import AEPsychStrategy, SequentialStrategy

from aepsych.server.message_handlers.handle_ask import handle_ask

logger = utils_logging.getLogger(logging.INFO)
DEFAULT_DESC = "default description"
DEFAULT_NAME = "default name"


def handle_get_config(server, request):
    msg = request["message"]
    section = msg.get("section", None)
    prop = msg.get("property", None)

    # If section and property are not specified, return the whole config
    if section is None and prop is None:
        return server.config.to_dict(deduplicate=False)

    # If section and property are not both specified, raise an error
    if section is None and prop is not None:
        raise RuntimeError("Message contains a property but not a section!")
    if section is not None and prop is None:
        raise RuntimeError("Message contains a section but not a property!")

    # If both section and property are specified, return only the relevant value from the config
    return server.config.to_dict(deduplicate=False)[section][prop]


def _configure(server, config):
    server._pregen_asks = (
        []
    )  # TODO: Allow each strategy to have its own stack of pre-generated asks

    parnames = config._str_to_list(config.get("common", "parnames"), element_type=str)
    server.parnames = parnames
    server.config = config
    server.use_ax = config.getboolean("common", "use_ax", fallback=False)
    server.enable_pregen = config.getboolean("common", "pregen_asks", fallback=False)
    if server.use_ax:
        server.trial_index = -1
        server.strat = AEPsychStrategy.from_config(config)
        server.strat_id = server.n_strats - 1

    else:
        server.strat = SequentialStrategy.from_config(config)
        server.strat_id = server.n_strats - 1  # 0-index strats

    return server.strat_id


def configure(server, config=None, **config_args):
    # To preserve backwards compatibility, config_args is still usable for unittests and old functions.
    # But if config is specified, the server will use that rather than create a new config object.
    if config is None:
        usedconfig = Config(**config_args)
    else:
        usedconfig = config
    if "experiment" in usedconfig:
        logger.warning(
            'The "experiment" section is being deprecated from configs. Please put everything in the "experiment" section in the "common" section instead.'
        )

        for i in usedconfig["experiment"]:
            usedconfig["common"][i] = usedconfig["experiment"][i]
        del usedconfig["experiment"]

    version = usedconfig.version
    if version < __version__:
        try:
            usedconfig.convert_to_latest()

            server.db.perform_updates()
            logger.warning(
                f"Config version {version} is less than AEPsych version {__version__}. The config was automatically modified to be compatible. Check the config table in the db to see the changes."
            )
        except RuntimeError:
            logger.warning(
                f"Config version {version} is less than AEPsych version {__version__}, but couldn't automatically update the config! Trying to configure the server anyway..."
            )

    server.db.record_config(master_table=server._db_master_record, config=usedconfig)
    return _configure(server, usedconfig)


def handle_setup(server, request):
    logger.debug("got setup message!")

    if not server.is_performing_replay:
        experiment_id = None
        if server._db_master_record is not None:
            experiment_id = server._db_master_record.experiment_id

        server._db_master_record = server.db.record_setup(
            description=DEFAULT_DESC,
            name=DEFAULT_NAME,
            request=request,
            id=experiment_id,
        )

    if (
        "config_str" in request["message"].keys()
        or "config_dict" in request["message"].keys()
    ):
        _ = configure(server, **request["message"])
    else:
        raise RuntimeError("Missing a configure message!")
    new_config = handle_ask(server, request)

    return new_config


def handle_setup_v01(server, request):
    logger.debug("got setup message!")
    ### make a temporary config object to derive parameters because server handles config after table
    if (
        "config_str" in request["message"].keys()
        or "config_dict" in request["message"].keys()
    ):
        tempconfig = Config(**request["message"])
        if not server.is_performing_replay:
            experiment_id = None
            if server._db_master_record is not None:
                experiment_id = server._db_master_record.experiment_id
            if "metadata" in tempconfig.keys():
                cdesc = (
                    tempconfig["metadata"]["experiment_description"]
                    if ("experiment_description" in tempconfig["metadata"].keys())
                    else DEFAULT_DESC
                )
                cname = (
                    tempconfig["metadata"]["experiment_name"]
                    if ("experiment_name" in tempconfig["metadata"].keys())
                    else DEFAULT_NAME
                )
                cid = (
                    tempconfig["metadata"]["experiment_id"]
                    if ("experiment_id" in tempconfig["metadata"].keys())
                    else None
                )
                server._db_master_record = server.db.record_setup(
                    description=cdesc,
                    name=cname,
                    request=request,
                    id=cid,
                    extra_metadata=tempconfig.jsonifyMetadata(),
                )
            ### if the metadata does not exist, we are going to log nothing
            else:
                server._db_master_record = server.db.record_setup(
                    description=DEFAULT_DESC,
                    name=DEFAULT_NAME,
                    request=request,
                    id=experiment_id,
                )

        strat_id = configure(server, config=tempconfig)
    else:
        raise RuntimeError("Missing a configure message!")

    return strat_id
