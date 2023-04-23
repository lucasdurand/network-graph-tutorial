# Part 0 - The Data

This whole section can be skimmed, but the goal here is to generate usable data that *looks* and *feels* like a real group of people

**Tasks**

* Use `Faker` to generate a population of sample people
* Use `numpy.random` and some sweet maths to fill in custom values
* Get all that into a `pd.DataFrame` and figure out a way to assign a manager to everyone
* Finally, hire a CEO and pay them too much

* Get started in the [**Workbook**](https://github.dev/lucasdurand/network-graph-tutorial/blob/develop/Part%200%20-%20The%20Data/Workbook.ipynb)

**Extras**

* Add more information to our people
* Align roles/locations more with reality:
    * Executives in headquarters location (with exceptions)
    * *Managers* should all be `"Full Time"`
* Merge logic back into `Faker` so that we can keep track of things like unique values and simplify the interface
* What other kinds of organizations can we model?