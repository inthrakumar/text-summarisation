class Counter:
    def __init__(self, iterable=None):
        self.counts = {}
        if iterable:
            self.update(iterable)

    def update(self, iterable):
        for element in iterable:
            if element in self.counts:
                self.counts[element] += 1
            else:
                self.counts[element] = 1

    def __getitem__(self, element):
        return self.counts.get(element, 0)

    def __setitem__(self, element, count):
        self.counts[element] = count

    def __delitem__(self, element):
        if element in self.counts:
            del self.counts[element]

    def __iter__(self):
        return iter(self.counts)

    def items(self):
        return self.counts.items()

    def most_common(self, n=None):
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts[:n] if n else sorted_counts
    
    def values(self):
        return list(self.counts.values())
    
    def keys(self):
        return self.counts.keys()

    def __repr__(self):
        return f"Counter({self.counts})"


