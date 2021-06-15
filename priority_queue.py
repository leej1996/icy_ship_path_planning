from queue import PriorityQueue


class CustomPriorityQueue(PriorityQueue):
    def _put(self, item):
        return super()._put((self._get_priority(item), item))  # prioritized based on f score

    def _get(self):
        return super()._get()[1]

    def _get_priority(self, item):
        return item[1]

    def _update(self, item, update_value):
        self.queue.remove(((item[1]), (item[0], item[1])))  # custom queue to update the priorities of objects
        self._put((item[0], update_value))
