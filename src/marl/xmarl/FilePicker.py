from textual.screen import Screen
from textual.widgets import ListView, ListItem, Label
from textual.containers import Vertical
from textual import events


class FilePickerScreen(Screen):
    BINDINGS = [("q", "app.pop_screen()", "Cancel")]

    def __init__(self, title: str, item_list: list[str]):
        super().__init__()
        self.title = title
        self.item_list = item_list

    def compose(self):
        yield Vertical(
            Label(f"{self.title}", id="dir_label"),
            ListView(*self.get_dir_items(), id="file_list"),
        )

    def get_dir_items(self):
        items = []

        # Add special entries for cancel
        items.append(ListItem(Label("[❌ Cancel]", id="cancel", markup=False)))

        # List directories and PDF files, sorted by date
        for entry in self.item_list:
            items.append(ListItem(Label(f"{entry}", markup=False)))
        return items

    # Wrap at edges of listview
    def on_key(self, event: events.Key) -> None:
        # only act if ListView has focus
        lv = self.query_one("#file_list", ListView)
        if not lv.has_focus:
            return

        # wrap at edges
        last = len(lv.children) - 1
        if event.key == "up":
            if lv.index == 0:
                lv.index = last
                event.stop()
        elif event.key == "down":
            if lv.index == last:
                lv.index = 0
                event.stop()

    def on_list_view_selected(self, event: ListView.Selected):
        label: Label = event.item.query_one(Label)
        text = str(label.renderable)

        # Handle "cancel" (exit picker)
        if text == "[❌ Cancel]":
            self.app.pop_screen()
            return

        self.dismiss(text)
