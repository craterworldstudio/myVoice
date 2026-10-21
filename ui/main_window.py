import customtkinter as ctk
from utils.config import config
try:
    from ui.stage_panel import StagePanel
except ImportError:
    from stage_panel import StagePanel





class MyVoiceGUI(ctk.CTk):

    def __init__(self, engine):
        super().__init__()

        self.engine = engine
        self.engine.build_pipeline()

        self.title("MyVoice DSP Control Panel")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_layout()

        self.after(50, self.update_gui)

    def on_close(self):
        try:
            self.engine.stop()
        except Exception:
            pass
        self.destroy()


    ###############################################################

    def build_layout(self):

        ###########################################################
        # Sidebar
        ###########################################################

        self.sidebar = ctk.CTkFrame(
            self,
            width=220,
            corner_radius=0
        )

        self.sidebar.pack(
            side="left",
            fill="y"
        )

        title = ctk.CTkLabel(
            self.sidebar,
            text="MyVoice",
            font=("Segoe UI", 26, "bold")
        )

        title.pack(
            pady=(25, 5)
        )

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Real-Time DSP Engine"
        )

        subtitle.pack(
            pady=(0, 20)
        )

        ###########################################################

        self.start_button = ctk.CTkButton(
            self.sidebar,
            text="Start Engine",
            command=self.start_engine
        )

        self.start_button.pack(
            fill="x",
            padx=10,
            pady=5
        )

        ###########################################################

        self.stop_button = ctk.CTkButton(
            self.sidebar,
            text="Stop Engine",
            command=self.stop_engine
        )

        self.stop_button.pack(
            fill="x",
            padx=10,
            pady=5
        )

        ###########################################################

        self.reload_button = ctk.CTkButton(
            self.sidebar,
            text="Reload Config",
            command=self.reload_config
        )

        self.reload_button.pack(
            fill="x",
            padx=10,
            pady=5
        )

        ###########################################################
        # Audio Devices & Hardware
        ###########################################################

        dev_label = ctk.CTkLabel(
            self.sidebar,
            text="Audio Devices",
            font=("Segoe UI", 14, "bold")
        )
        dev_label.pack(anchor="w", padx=12, pady=(15, 2))

        import sounddevice as sd
        in_devs = ["pulse", "default"]
        out_devs = ["pulse", "default"]
        try:
            for d in sd.query_devices():
                n = d['name']
                if d['max_input_channels'] > 0 and n not in in_devs:
                    in_devs.append(n)
                if d['max_output_channels'] > 0 and n not in out_devs:
                    out_devs.append(n)
        except Exception:
            pass

        in_lbl = ctk.CTkLabel(self.sidebar, text="Mic / Input:", anchor="w")
        in_lbl.pack(fill="x", padx=12, pady=(2, 0))

        cur_in = getattr(config.device, "input", "pulse") or "pulse"
        if cur_in not in in_devs:
            in_devs.insert(0, cur_in)
        self.in_option = ctk.CTkOptionMenu(
            self.sidebar,
            values=in_devs[:8],
            command=self.on_input_device_change
        )
        self.in_option.set(cur_in)
        self.in_option.pack(fill="x", padx=10, pady=(0, 6))

        out_lbl = ctk.CTkLabel(self.sidebar, text="Speaker / Output:", anchor="w")
        out_lbl.pack(fill="x", padx=12, pady=(2, 0))

        cur_out = getattr(config.device, "output", "pulse") or "pulse"
        if cur_out not in out_devs:
            out_devs.insert(0, cur_out)
        self.out_option = ctk.CTkOptionMenu(
            self.sidebar,
            values=out_devs[:8],
            command=self.on_output_device_change
        )
        self.out_option.set(cur_out)
        self.out_option.pack(fill="x", padx=10, pady=(0, 10))

        gpu_box = ctk.CTkFrame(self.sidebar, fg_color="#1e293b", corner_radius=6)
        gpu_box.pack(fill="x", padx=10, pady=(10, 5))
        gpu_lbl = ctk.CTkLabel(
            gpu_box,
            text="GPU: AMD Radeon RX 9060 XT\nROCm 7.2 Active",
            font=("Segoe UI", 11),
            text_color="#38bdf8",
            justify="center"
        )
        gpu_lbl.pack(padx=6, pady=6)


        ###########################################################
        # Main Panel
        ###########################################################

        self.main = ctk.CTkFrame(self)

        self.main.pack(
            side="left",
            fill="both",
            expand=True
        )

        ###########################################################

        self.stage_panel = StagePanel(
            self.main,
            self.engine
        )

        self.stage_panel.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        ###########################################################
        # Status Bar
        ###########################################################

        self.status = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w",
            height=30
        )

        self.status.pack(
            side="bottom",
            fill="x"
        )

    ###############################################################

    def start_engine(self):

        try:
            self.engine.start()
            self.status.configure(
                text="Engine Running"
            )

        except Exception as e:

            self.status.configure(
                text=str(e)
            )

    ###############################################################

    def stop_engine(self):

        try:
            self.engine.stop()

            self.status.configure(
                text="Engine Stopped"
            )

        except Exception as e:

            self.status.configure(
                text=str(e)
            )

    ###############################################################

    def reload_config(self):

        try:

            self.engine.reload_config()

            self.stage_panel.refresh()

            self.status.configure(
                text="Configuration Reloaded"
            )

        except Exception as e:

            self.status.configure(
                text=str(e)
            )

    ###############################################################

    def update_gui(self):

        self.stage_panel.update_values()

        self.after(
            50,
            self.update_gui
        )

    def on_input_device_change(self, value):
        from utils.config import config
        config.device.input = value
        self.status.configure(text=f"Input device set to {value} (restarts when engine restarted)")

    def on_output_device_change(self, value):
        from utils.config import config
        config.device.output = value
        self.status.configure(text=f"Output device set to {value} (restarts when engine restarted)")