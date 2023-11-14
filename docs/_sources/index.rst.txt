.. Computational Software documentation master file, created by
   sphinx-quickstart on Tue Mar 31 12:45:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. figure:: img/CompSoft_banner.png

Willkommen zum Kurs Einführung in die Modellierung Physikalischer Prozesse!
========================================================================

Die Programmiersprache Python ist für alle Arten von wissenschaftlichen und technischen Aufgaben nützlich. Sie können mit ihr Daten analysieren und darstellen. Sie können mit ihr auch wissenschaftliche Probleme numerisch lösen, die analytisch nur schwer oder gar nicht zu lösen sind. Python ist frei verfügbar und wurde aufgrund seines modularen Aufbaus um eine nahezu unendliche Anzahl von Modulen für verschiedene Zwecke erweitert.

Dieser Kurs soll Sie in die Programmierung mit Python einführen. Er richtet sich eher an den Anfänger, wir hoffen aber, dass er auch für Fortgeschrittene interessant ist.
Wir beginnen den Kurs mit einer Einführung in die Jupyter Notebook-Umgebung, die wir während des gesamten Kurses verwenden werden. Danach werden wir eine Einführung in Python geben und Ihnen einige grundlegende Funktionen zeigen, wie z.B. das Plotten und Analysieren von Daten durch Kurvenanpassung, das Lesen und Schreiben von Dateien, was einige der Aufgaben sind, die Ihnen während Ihres Physikstudiums begegnen werden. Wir zeigen Ihnen auch einige fortgeschrittene Themen wie die Animation in Jupyter und die Simulation von physikalischen Prozessen in

* Mechanik
* Elektrostatik
* Wellen
* Optik

Falls am Ende des Kurses Zeit bleibt, werden wir auch einen Blick auf Verfahren des **maschinellen Lernens** werfen, das mittlerweile auch in der Physik zu einem wichtigen Werkzeug geworden ist.

Wir werden keine umfassende Liste von numerischen Simulationsschemata präsentieren, sondern die Beispiele nutzen, um Ihre Neugierde zu wecken. Da es leichte Unterschiede in der Syntax der verschiedenen Python-Versionen gibt, werden wir uns im Folgenden immer auf den **Python 3**-Standard beziehen.

Der Kurs wird auf Deutsch gehalten werden. Die Webseiten, die Sie für den Überblick zu Python zur Verfügung gestellt bekommen, werden allerdings auf Englisch sein. Übungsaufgaben werden werden auf Deutsch gestellt.

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/lSIwZFeRpfQ" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

|
|

.. toctree::
   :maxdepth: 2
   :caption: Informationen:

   course-info/website.rst
   course-info/schedule.rst
   course-info/assignments.rst
   course-info/exam.rst
   course-info/resources.rst
   course-info/instructor.rst

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   lectures/Intro/overview.rst
   notebooks/Intro/1_Introduction2Jupyter.ipynb
   notebooks/Intro/2_NotebookEditor.ipynb
   notebooks/Intro/3_EditCells.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Lecture 1:

   lectures/L1/overview_1.rst
   notebooks/L1/1_variables.ipynb
   notebooks/L1/2_operators.ipynb


.. toctree::
   :maxdepth: 2
   :caption: Lecture 2:
   
   lectures/L2/overview_2.rst
   notebooks/L1/3_datatypes.ipynb
   notebooks/L1/4_modules.ipynb
   notebooks/L2/1_numpy.ipynb   
   notebooks/L2/2_plotting.ipynb
   lectures/L2/assignment_1.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Lecture 3:

   lectures/L3/overview_3.rst
   notebooks/L2/3_randomnumbers.ipynb
   notebooks/L3/1_input_output.ipynb
   notebooks/L3/2_flowcontrol.ipynb   
   notebooks/L3/4_exceptions.ipynb
   lectures/L3/assignment_2.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
