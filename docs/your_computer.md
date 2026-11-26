# On your own computer

Run Petrophys on your own computer when your well data must not leave it.
You install three things once: Python, Petrophys, and JupyterLab, the program
that opens notebooks. It takes about ten minutes. If notebooks are new to you,
read {doc}`new_to_python` first.

Your data stays on your computer. JupyterLab runs on your computer and shows
its notebooks in your web browser at an address that starts with
`localhost`, which means your own machine. Petrophys reads and writes only
the files you name and sends nothing over the network. The one download is
the 112 MB borehole-image sample that the borehole-image tutorial fetches on
first use.

On a company computer you may not be allowed to install programs. Ask your IT
department for Python 3.13 and permission to install Python packages with
`pip`, or for your company's package mirror.

## 1. Install Python

Petrophys runs on Python 3.10 to 3.13. The main download button on python.org
offers a later version, so take a Python 3.13 installer from the page for your
system:

- **Windows:** open [python.org/downloads/windows](https://www.python.org/downloads/windows/),
  find the newest **Python 3.13** release in the list of stable releases and
  download its **Windows installer (64-bit)**. Run it: on the first screen,
  tick **Add python.exe to PATH**, then click **Install Now**.
- **macOS:** open [python.org/downloads/macos](https://www.python.org/downloads/macos/),
  find the newest **Python 3.13** release and download its **macOS
  installer**. Open the downloaded `.pkg` file and follow the steps.

## 2. Install Petrophys and JupyterLab

Open a terminal, a window where you type commands:

- **Windows:** open the Start menu, type `cmd` and open **Command Prompt**.
- **macOS:** open **Terminal** (Applications > Utilities).

Type this command and press Enter:

::::{tab-set}
:::{tab-item} Windows
```bat
py -m pip install "petrophys[ml]" jupyterlab
```
:::
:::{tab-item} macOS
```bash
python3 -m pip install "petrophys[ml]" jupyterlab
```
:::
::::

It downloads and installs for a minute or two. `[ml]` adds the machine
learning used to fill gaps in logs.

## 3. Make a folder for your work

Create a folder, for example `Documents\petrophys`, and copy your LAS files
into it. To work through a tutorial, open its page on this site, click the
eye icon (**View this page**) at the top of the page to see the notebook on
GitHub, and use GitHub's **Download raw file** button to save the `.ipynb`
file into the same folder.

## 4. Start JupyterLab

In the terminal, go to your folder and start JupyterLab:

::::{tab-set}
:::{tab-item} Windows
```bat
cd %USERPROFILE%\Documents\petrophys
py -m jupyterlab
```
:::
:::{tab-item} macOS
```bash
cd ~/Documents/petrophys
python3 -m jupyterlab
```
:::
::::

JupyterLab opens in your web browser and lists the files in the folder.
Keep the terminal window open while you work: it is JupyterLab running.

## 5. Open a notebook and read your well

Double-click a downloaded tutorial to open it, or start a new one with
**File > New > Notebook** and choose **Python 3**. In the first cell, type and
run (Shift + Enter):

```python
import petrophys as pp

well = pp.read("my_well.las")   # a LAS file in your folder
pp.plot(well).show()
```

`pp.read("logs")` reads every LAS file in a `logs` folder at once. Results
you save, such as `result.to_las("my_well_interpreted.las")`, are written to
the same folder.

## Next time

Open a terminal, go to your folder and start JupyterLab again (step 4). To
stop, choose **File > Shut Down** in JupyterLab, then close the terminal.

To update Petrophys, run `py -m pip install --upgrade petrophys` (Windows) or
`python3 -m pip install --upgrade petrophys` (macOS).

If you use VS Code, install its Python and Jupyter extensions and open the
notebooks there instead of JupyterLab. Anaconda also works; check your
company's licence first, since Anaconda's terms require a paid licence for
organizations of 200 or more employees.

## Next

{doc}`quickstart`: read, plot, interpret and export a first well.
