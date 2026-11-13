# New to Python?

You do not need to have programmed before to use Petrophys. This page explains
the few things you need: what a notebook is, how to run it, and how to read
your own well file.

Petrophys is a Python *library*: a set of ready-made tools you use by typing
short commands such as `pp.read("my_well.las")` or `pp.plot(well)`. You type
them in a *notebook*.

## What a notebook is

A notebook is a document made of *cells*. Text cells explain; code cells hold a
few lines of Python. When you run a code cell, its result (a number, a table,
a log plot) appears right below it. The {doc}`quickstart` and every
{doc}`tutorial <tutorials/index>` on this site are notebooks.

## Two ways to run it

| | In your browser (Google Colab) | On your own computer |
| --- | --- | --- |
| What you install | Nothing: a web browser and a Google account | Python, Petrophys and JupyterLab, once (about 10 minutes) |
| Data | The sample wells that come with Petrophys, and files you upload | Any file on your computer |
| Where your data goes | Google's servers | Nowhere: it stays on your computer |
| Best for | Trying Petrophys out | Company and client data |

Start in your browser. Before you upload a company well to Colab, check your
company's rules on cloud services; for confidential data, use
{doc}`your own computer <your_computer>`.

## Open a notebook in Colab

1. On the {doc}`quickstart` or any tutorial page, click
   ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg).
2. Sign in to Google if asked. Colab opens your own copy of the notebook;
   nothing you change affects anyone else.
3. Run the first code cell. It installs Petrophys, which takes about a
   minute. Colab first warns that the notebook was not written by Google:
   click **Run anyway**.

## Run a cell

- **Run one cell:** click inside it and press **Shift + Enter**, or click the
  ▶ button at its left. The result appears below the cell, and the next cell
  is selected, so pressing Shift + Enter again walks down the notebook.
- **Run cells from the top down.** A cell uses what the cells above it made.
  If you see `NameError: name 'well' is not defined`, a cell above it has not
  run yet. In Colab, **Runtime > Run before** runs every cell above the
  selected one.
- **Wait for a running cell.** While a cell runs, its ▶ button spins
  (Colab) or its label shows `[*]` (JupyterLab).
- **Try your own values.** Change a number, such as `top=2800` to
  `top=2900`, and run the cell again. That is how you try other settings.
- **Add a cell:** click **+ Code**, type, and run it.
- **Start over:** **Runtime > Restart session** in Colab, or
  **Kernel > Restart Kernel** in JupyterLab. Python forgets everything it
  ran, so run the cells from the top again.
- **Read an error from its last line.** An error appears in red below the
  cell, and its last line says what went wrong, for example
  `FileNotFoundError` when a file name is misspelled.

## Read your own well file in Colab

1. Click the folder icon in the left sidebar to open the **Files** panel.
2. Click the upload icon at the top of the panel and choose your LAS file, or
   drag the file into the panel.
3. Add a code cell and run:

   ```python
   import petrophys as pp

   well = pp.read("my_well.las")   # the name of the file you uploaded
   print(well)
   pp.plot(well).show()
   ```

4. Interpret it and save the result as a LAS file:

   ```python
   result = pp.interpret(well)
   result.to_las("my_well_interpreted.las")
   ```

   The new file appears in the **Files** panel (click the refresh icon if it
   does not). Right-click it and choose **Download**.

Colab deletes uploaded and saved files when the session ends, so download
what you want to keep. To read many wells, put them in a folder on Google
Drive, connect the drive and read the whole folder:

```python
from google.colab import drive

drive.mount("/content/drive")                       # asks for your permission
wells = pp.read("/content/drive/MyDrive/my_wells")  # every LAS file in the folder
```

## Words you will meet

| Word | Meaning |
| --- | --- |
| Python | The programming language Petrophys is written in. |
| Library | A package of ready-made tools; Petrophys is one. |
| `import petrophys as pp` | Loads Petrophys and gives it the short name `pp`, so commands read `pp.read`, `pp.plot`. |
| Function | A named command that takes inputs in brackets: `pp.read("my_well.las")`. |
| Variable | A name for a result, set with `=`: after `well = pp.read(...)`, `well` is the well. |
| Session (kernel) | The Python running behind a notebook. Restarting it clears every variable. |
| `pip install` | The command that installs a Python library. |

## Next

{doc}`quickstart`: read, plot, interpret and export a first well, in your
browser or on your computer.
