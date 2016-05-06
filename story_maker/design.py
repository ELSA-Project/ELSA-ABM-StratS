# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test_interface_template.ui'
#
# Created: Fri May  6 17:44:40 2016
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1528, 817)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.main_splitter = QtGui.QSplitter(self.centralwidget)
        self.main_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.main_splitter.setObjectName(_fromUtf8("main_splitter"))
        self.verticalLayoutWidget_2 = QtGui.QWidget(self.main_splitter)
        self.verticalLayoutWidget_2.setObjectName(_fromUtf8("verticalLayoutWidget_2"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.main_frame = QtGui.QWidget(self.verticalLayoutWidget_2)
        self.main_frame.setObjectName(_fromUtf8("main_frame"))
        self.verticalLayout_4.addWidget(self.main_frame)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.pause = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.pause.setObjectName(_fromUtf8("pause"))
        self.horizontalLayout_2.addWidget(self.pause)
        self.play = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.play.setObjectName(_fromUtf8("play"))
        self.horizontalLayout_2.addWidget(self.play)
        self.play_one_step = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.play_one_step.setObjectName(_fromUtf8("play_one_step"))
        self.horizontalLayout_2.addWidget(self.play_one_step)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.text_splitter = QtGui.QSplitter(self.main_splitter)
        self.text_splitter.setOrientation(QtCore.Qt.Vertical)
        self.text_splitter.setObjectName(_fromUtf8("text_splitter"))
        self.story = QtGui.QTextBrowser(self.text_splitter)
        self.story.setObjectName(_fromUtf8("story"))
        self.information = QtGui.QTextBrowser(self.text_splitter)
        self.information.setObjectName(_fromUtf8("information"))
        self.horizontalLayout.addWidget(self.main_splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1528, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pause.setText(_translate("MainWindow", "Pause", None))
        self.play.setText(_translate("MainWindow", "Play", None))
        self.play_one_step.setText(_translate("MainWindow", "Play One step", None))

