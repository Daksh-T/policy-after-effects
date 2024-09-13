import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import textwrap

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Tree, Static, Label
from textual import events
from rich.text import Text
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
executor = ThreadPoolExecutor()

class PolicyApp(App):
    def __init__(self, policy_file=None, **kwargs):
        super().__init__(**kwargs)
        self.policy_file = policy_file
        self.input_dialog_submitted = False
        self.input_dialog_value = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Enter policy here:")
        self.policy_input = Input(id="policy_input", placeholder="Enter policy here...")
        yield self.policy_input

        self.effects_tree = Tree("Effects", id="effects_tree")
        self.effects_tree.root.data = {'expanded': True, 'order': 0}
        yield self.effects_tree

        self.response_display = Static(id="response_display")
        self.response_display.styles.width = "100%"
        self.response_display.styles.overflow = "auto"
        self.response_display.visible = False
        yield self.response_display

        yield Footer()

    def on_mount(self):
        self.policy_input.focus()
        if self.policy_file:
            asyncio.create_task(self.load_policy_from_file(self.policy_file))

    async def load_policy_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                policy = file.read()
            await self.generate_and_display_effects(policy, order=1)
            self.effects_tree.focus()
        except Exception as e:
            self.response_display.update(Text(f"Error reading file: {e}", style="bold red"))
            self.response_display.visible = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "policy_input":
            policy = event.value
            asyncio.create_task(self.generate_and_display_effects(policy, order=1))
            self.effects_tree.focus()
        elif event.input.id == "user_input":
            self.input_dialog_value = event.value
            self.input_dialog_submitted = True

    def get_color_for_order(self, order):
        colors = ["green", "yellow", "cyan", "magenta", "blue"]
        return colors[order % len(colors)]

    async def generate_and_display_effects(self, text, order, node=None):
        effects = await self.get_effects(text, order)
        if node is None:
            self.effects_tree.root.label = "Effects"
            self.effects_tree.root.remove_children()
            self.effects_tree.root.data = {'expanded': True, 'order': 0, 'full_text': "Root"}
            parent = self.effects_tree.root
        else:
            parent = node

        color = self.get_color_for_order(order)

        for effect in effects:
            # Wrap the text to 80 characters
            wrapped_lines = textwrap.wrap(effect['title'], width=80)
            
            label_text = Text(f"• {wrapped_lines[0]}", style=f"bold {color}")
            effect_node = parent.add(label_text, data={
                'expanded': False, 
                'order': order,
                'full_text': effect['title']
            })
            
            for line in wrapped_lines[1:]:
                label_text = Text(f"  {line}", style=color)
                effect_node.add(label_text, data={
                    'expanded': False,
                    'order': order,
                    'full_text': effect['title']
                })

        if node is None:
            self.effects_tree.root.expand()

    async def get_effects(self, text, order):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.sync_get_effects, text, order)

    def sync_get_effects(self, text, order):
        try:
            prompt = f"List the {order}-order effects of the following policy:\n\n{text}"
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            response = chat_completion.choices[0].message.content
            return self.parse_effects(response)
        except Exception as e:
            return [{'title': f"Error fetching effects: {e}"}]

    def parse_effects(self, response):
        effects = []
        for line in response.strip().split('\n'):
            if line.strip():
                effects.append({'title': line.strip()})
        return effects

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node
        if node.data and not node.data.get('expanded'):
            node.data['expanded'] = True
            current_order = node.data.get('order', 1)
            next_order = current_order + 1
            
            full_text = node.data.get('full_text', node.label.plain)
            
            asyncio.create_task(
                self.generate_and_display_effects(full_text, order=next_order, node=node)
            )

    async def prompt_follow_up_question(self, node):
        self.input_dialog_submitted = False
        
        full_text = node.data.get('full_text', node.label.plain)
        
        question = await self.run_input_dialog(f"Question about '{full_text}':")
        if question:
            response = await self.ask_groq(question)
            self.response_display.update(Text(response, no_wrap=False, overflow="fold"))
            self.response_display.visible = True


    async def run_input_dialog(self, prompt_text):
        prompt_label = Label(prompt_text)
        await self.mount(prompt_label)

        input_widget = Input(id="user_input", placeholder="Enter your input...")
        await self.mount(input_widget)
        input_widget.focus()

        self.input_dialog_submitted = False
        self.input_dialog_value = None
        while not self.input_dialog_submitted:
            await asyncio.sleep(0.1)

        value = self.input_dialog_value

        await input_widget.remove()
        await prompt_label.remove()

        return value

    async def prompt_file_path(self):
        self.input_dialog_submitted = False
        file_path = await self.run_input_dialog("Enter the path to the policy text file:")
        if file_path:
            await self.load_policy_from_file(file_path)

    async def ask_groq(self, question):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.sync_ask_groq, question)

    def sync_ask_groq(self, question):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error fetching response: {e}"

    def on_key(self, event: events.Key) -> None:
        if event.key == "right":
            node = self.effects_tree.cursor_node
            if node:
                asyncio.create_task(self.prompt_follow_up_question(node))
        elif event.key == "u":
            asyncio.create_task(self.prompt_file_path())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Policy Effects TUI Application")
    parser.add_argument('--file', type=str, help='Path to policy text file')
    args = parser.parse_args()

    app = PolicyApp(policy_file=args.file)
    app.run()