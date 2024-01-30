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
   lectures/L3/assignment_2.rst
   notebooks/L3/25_publication_ready_figures.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Lecture 4:

   lectures/L4/overview_4.rst
   notebooks/L3/1_input_output.ipynb
   notebooks/L3/2_flowcontrol.ipynb   
   notebooks/L3/3_functions.ipynb   
   lectures/L4/assignment_3.rst

.. toctree::
   :maxdepth: 2
   :caption: Lecture 4 Self Study:

   notebooks/L3/4_exceptions.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Lecture 5:

   lectures/L4/overview_5.rst
   notebooks/L4/1_classes.ipynb
   notebooks/L4/2_brownian_motion.ipynb   
   notebooks/L4/3_animations.ipynb  
   lectures/L5/assignment_4.rst


.. toctree::
   :maxdepth: 2
   :caption: Lecture 6:

   lectures/L5/overview_5.rst
   notebooks/L5/1_differentiation.ipynb
   notebooks/L5/2_integration.ipynb   
   notebooks/L5/3_solving_ODEs.ipynb 
   lectures/L6/assignment_5.rst

.. toctree::
   :maxdepth: 2
   :caption: Lecture 7:

   lectures/L6/overview_6.rst   
   notebooks/L6/2_coupled_pendula.ipynb   
   notebooks/L6/3_fourier_analysis.ipynb  

.. toctree::
   :maxdepth: 2
   :caption: Lecture 7 Self Study:
   
   notebooks/L6/1_covid19.ipynb
   
.. toctree::
   :maxdepth: 2
   :caption: Lecture 7 Seminar Coding:
   
   notebooks/L6/4_fourier_series_coding.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Lecture 8:

   notebooks/L7/1_spring_pendulum.ipynb
   notebooks/L7/2_planetary_motion.ipynb   
      
   
.. toctree::
   :maxdepth: 2
   :caption: Lecture 9:

   notebooks/L7/3_diffusion_equation.ipynb     
   notebooks/L8/1_curve_fitting.ipynb


.. toctree::
   :maxdepth: 2
   :caption: Lecture 10:

   notebooks/L8/1_curve_fitting.ipynb
   notebooks/L9/1_plane_waves.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Extra Seminar:

   notebooks/L12/2_reinforcement_learning.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Lecture 11:

   notebooks/L9/1_plane_waves.ipynb
   notebooks/L9/2_spherical_waves.ipynb
   notebooks/L9/3_huygens_principle.ipynb
   
.. toctree::
   :maxdepth: 2
   :caption: Lecture 12:

   notebooks/L13/1_deep_learning.ipynb
   
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
