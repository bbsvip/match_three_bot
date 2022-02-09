""" Created by MrBBS """
# 2/8/2022
# -*-encoding:utf-8-*-

from random import random
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.stacklayout import StackLayout
from kivy.clock import Clock
from kivy.uix.slider import Slider
from kivy.core.window import Window
from kivy.app import App

thickness = 5


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        color = (random(), 1, 1)
        with self.canvas:
            Color(*color, mode='hsv')
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=thickness)

    def on_touch_move(self, touch):
        if 'line' in touch.ud:
            touch.ud['line'].points += [touch.x, touch.y]


class MyPaintApp(App):

    def build(self):
        # layouts
        w = Widget()
        b = BoxLayout()
        f = FloatLayout()
        a = AnchorLayout(anchor_x='left', anchor_y='bottom')
        s = StackLayout()
        # widgets
        p = MyPaintWidget()
        cb = Button(text='Clear')
        lb = Button(pos=(150, 0), text="Large\nBrush", halign="center")
        mb = Button(pos=(300, 0), text="Medium\nBrush", halign="center")
        sb = Button(pos=(450, 0), text="Small\nBrush", halign="center")
        svb = Button(pos=(600, 0), text="Save", halign="center")

        # adding the widgets
        w.add_widget(p)
        w.add_widget(cb)
        w.add_widget(lb)
        w.add_widget(mb)
        w.add_widget(sb)
        w.add_widget(svb)

        # w.export_to_png('ScreenShot.png')

        # bindings
        def clear_canvas(obj):
            p.canvas.clear()

        cb.bind(on_release=clear_canvas)

        def large_brush(obj):
            global thickness
            thickness = 15

        lb.bind(on_release=large_brush)

        def medium_brush(obj):
            global thickness
            thickness = 10

        mb.bind(on_release=medium_brush)

        def small_brush(obj):
            global thickness
            thickness = 5

        sb.bind(on_release=small_brush)

        def Screenshot(obj):
            global Screenshotnum
            Window.screenshot(name="Screenshot.png")

        svb.bind(on_release=Screenshot)

        return w


if __name__ == '__main__':
    MyPaintApp().run()
