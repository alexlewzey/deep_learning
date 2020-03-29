"""a module of functions that are useful when used with the pytorch module"""


def show_train_stats(epoch: int, batch: int, loss: float, accuracy: float, frequency: int = 500) -> None:
    if batch % frequency == 0:
        msg: str = f'epoch: {epoch:2}, batch: {batch:5}, loss: {loss:.2f}, accuracy: {accuracy:.2f}'
        print(msg)
