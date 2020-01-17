import os

__all__ = ["is_using_hvd"]


def is_using_hvd():
    env_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]

    if all([var in os.environ for var in env_vars]):
        return True
    else:
        return False
