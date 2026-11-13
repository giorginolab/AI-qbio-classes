import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(plt):
    plt.plot([1,2,3])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
