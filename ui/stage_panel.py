import os
from tkinter import filedialog
import customtkinter as ctk

def discover_rvc_models():
    found = {}
    base_dir = "./models"
    if os.path.isdir(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for f in sorted(files):
                if f.endswith(".pth") and not f.startswith("."):
                    full_path = os.path.abspath(os.path.join(root, f))
                    name = os.path.splitext(f)[0]
                    if "xiaomu" in name.lower():
                        display_name = f"{name} (Female)"
                    else:
                        display_name = name
                    found[display_name] = full_path
    return found

class StageFrame(ctk.CTkFrame):
    """
    Reusable collapsible frame for a DSP stage.
    """

    def __init__(self, master, title, stage=None, on_toggle_enable=None):
        super().__init__(master)

        self.stage = stage
        self.expanded = True

        self.columnconfigure(0, weight=1)

        #######################################################

        self.header = ctk.CTkFrame(self)

        self.header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=5,
            pady=5
        )

        self.header.columnconfigure(1, weight=1)

        #######################################################

        self.toggle_button = ctk.CTkButton(
            self.header,
            width=35,
            text="▼",
            command=self.toggle
        )

        self.toggle_button.grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )

        #######################################################

        self.title = ctk.CTkLabel(
            self.header,
            text=title,
            font=("Segoe UI", 16, "bold")
        )

        self.title.grid(
            row=0,
            column=1,
            sticky="w",
            padx=5
        )

        if stage is not None and on_toggle_enable is not None:
            self.enable_var = ctk.BooleanVar(value=getattr(stage, "enabled", False))
            self.enable_cb = ctk.CTkCheckBox(
                self.header,
                text="Enable",
                variable=self.enable_var,
                command=lambda: on_toggle_enable(self.enable_var.get())
            )
            self.enable_cb.grid(
                row=0,
                column=2,
                padx=10,
                pady=5
            )


        #######################################################

        self.content = ctk.CTkFrame(self)

        self.content.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=10,
            pady=(0, 10)
        )

        self.content.columnconfigure(1, weight=1)

    ###########################################################

    def toggle(self):

        if self.expanded:

            self.content.grid_remove()

            self.toggle_button.configure(text="▶")

            self.expanded = False

        else:

            self.content.grid()

            self.toggle_button.configure(text="▼")

            self.expanded = True


class SliderRow(ctk.CTkFrame):

    def __init__(
        self,
        master,
        text,
        minimum,
        maximum,
        value,
        callback
    ):

        super().__init__(master)

        self.columnconfigure(1, weight=1)

        ###################################################

        self.label = ctk.CTkLabel(
            self,
            text=text,
            width=110
        )

        self.label.grid(
            row=0,
            column=0,
            padx=5,
            pady=5,
            sticky="w"
        )

        ###################################################

        self.slider = ctk.CTkSlider(
            self,
            from_=minimum,
            to=maximum,
            command=self.changed
        )

        self.slider.set(value)

        self.slider.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=5
        )

        ###################################################

        self.value = ctk.CTkLabel(
            self,
            width=70,
            text=f"{value:.3f}"
        )

        self.value.grid(
            row=0,
            column=2,
            padx=5
        )

        ###################################################

        self.callback = callback

    #######################################################

    def changed(self, value):

        self.value.configure(
            text=f"{value:.3f}"
        )

        if self.callback:
            self.callback(value)

    def set_value(self, value):
        self.slider.set(value)
        self.value.configure(
            text=f"{value:.3f}"
        )
        if self.callback:
            self.callback(value)

class ToggleRow(ctk.CTkFrame):


    def __init__(
        self,
        master,
        text,
        value,
        callback
    ):

        super().__init__(master)

        self.variable = ctk.BooleanVar(
            value=value
        )

        self.checkbox = ctk.CTkCheckBox(
            self,
            text=text,
            variable=self.variable,
            command=self.changed
        )

        self.checkbox.pack(
            anchor="w",
            padx=5,
            pady=5
        )

        self.callback = callback

    def changed(self):

        if self.callback:

            self.callback(
                self.variable.get()
            )

class OptionRow(ctk.CTkFrame):

    def __init__(
        self,
        master,
        text,
        values,
        current,
        callback
    ):

        super().__init__(master)

        self.columnconfigure(1, weight=1)

        ##################################################

        self.label = ctk.CTkLabel(
            self,
            text=text,
            width=100
        )

        self.label.grid(
            row=0,
            column=0,
            padx=5
        )

        ##################################################

        self.option = ctk.CTkOptionMenu(
            self,
            values=values,
            command=callback
        )

        self.option.set(current)

        self.option.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=5,
            pady=5
        )


class StagePanel(ctk.CTkScrollableFrame):

    def __init__(
        self,
        master,
        engine
    ):

        super().__init__(master)

        self.engine = engine

        self.frames = {}

        self.build()

    ########################################################

    def build(self):
        # 0. Profile Selector Header
        profile_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray17"), corner_radius=8)
        profile_frame.pack(fill="x", padx=10, pady=(6, 12))

        prof_title = ctk.CTkLabel(
            profile_frame,
            text="✨ VOICE PROFILES",
            font=("Segoe UI", 15, "bold"),
            text_color="#3B8ED0"
        )
        prof_title.pack(anchor="w", padx=12, pady=(8, 4))

        btn_bar = ctk.CTkFrame(profile_frame, fg_color="transparent")
        btn_bar.pack(fill="x", padx=8, pady=(0, 10))

        def activate_profile(prof_name):
            self.engine.set_profile(prof_name)
            # Update checkboxes visually
            if "formant_shifter" in self.frames and hasattr(self.frames["formant_shifter"], "enable_var"):
                self.frames["formant_shifter"].enable_var.set(prof_name == "female")
            if "radio_effect" in self.frames and hasattr(self.frames["radio_effect"], "enable_var"):
                self.frames["radio_effect"].enable_var.set(prof_name == "alastor")
            if "rvc" in self.frames and hasattr(self.frames["rvc"], "enable_var"):
                self.frames["rvc"].enable_var.set(prof_name == "rvc")

        btn_alastor = ctk.CTkButton(
            btn_bar,
            text="📻 Alastor (Radio Demon)",
            height=34,
            fg_color="#8B1E1E",
            hover_color="#B22222",
            font=("Segoe UI", 13, "bold"),
            command=lambda: activate_profile("alastor")
        )
        btn_alastor.pack(side="left", fill="x", expand=True, padx=4)

        btn_female = ctk.CTkButton(
            btn_bar,
            text="👩 Female Voice (Real-Time Formant)",
            height=34,
            fg_color="#7B1FA2",
            hover_color="#9C27B0",
            font=("Segoe UI", 13, "bold"),
            command=lambda: activate_profile("female")
        )
        btn_female.pack(side="left", fill="x", expand=True, padx=4)

        btn_clean = ctk.CTkButton(
            btn_bar,
            text="🎙️ Bypass (Clean)",
            height=34,
            fg_color="#37474F",
            hover_color="#455A64",
            font=("Segoe UI", 13, "bold"),
            command=lambda: activate_profile("clean")
        )
        btn_clean.pack(side="left", fill="x", expand=True, padx=4)

        # Formant Shifter Stage (Option 2: Ultra-low latency Male-to-Female)
        formant = self.engine.get_stage("formant_shifter")
        if formant:
            ff = StageFrame(
                self,
                "Female Formant & Pitch Shifter (Real-Time DSP)",
                stage=formant,
                on_toggle_enable=lambda val: self.toggle_stage("formant_shifter", val)
            )
            ff.pack(fill="x", padx=10, pady=6)
            self.frames["formant_shifter"] = ff

            s_fpitch = SliderRow(
                ff.content,
                "Pitch Shift (Semitones)",
                minimum=0.0,
                maximum=12.0,
                value=float(getattr(formant, "pitch_semitones", 4.2)),
                callback=lambda val: formant.set_pitch(val)
            )
            s_fpitch.pack(fill="x", padx=5, pady=3)

            s_fformant = SliderRow(
                ff.content,
                "Formant / Vocal Tract Shift",
                minimum=0.0,
                maximum=8.0,
                value=float(getattr(formant, "formant_semitones", 3.0)),
                callback=lambda val: formant.set_formant(val)
            )
            s_fformant.pack(fill="x", padx=5, pady=3)

            # Quick Vocal Tuning Presets
            f_preset_row = ctk.CTkFrame(ff.content, fg_color="transparent")
            f_preset_row.pack(fill="x", padx=5, pady=(4, 6))

            f_lbl = ctk.CTkLabel(f_preset_row, text="Presets:", width=65, anchor="w")
            f_lbl.pack(side="left", padx=5)

            def set_tuning(p, f):
                s_fpitch.set_value(p)
                s_fformant.set_value(f)

            btn_f_nat = ctk.CTkButton(
                f_preset_row,
                text="Natural (+5.0, +3.5)",
                width=120,
                height=24,
                command=lambda: set_tuning(5.0, 3.5)
            )
            btn_f_nat.pack(side="left", padx=3)

            btn_f_bright = ctk.CTkButton(
                f_preset_row,
                text="Bright (+6.0, +4.2)",
                width=115,
                height=24,
                command=lambda: set_tuning(6.0, 4.2)
            )
            btn_f_bright.pack(side="left", padx=3)

            btn_f_deep = ctk.CTkButton(
                f_preset_row,
                text="Warm (+4.0, +2.8)",
                width=110,
                height=24,
                command=lambda: set_tuning(4.0, 2.8)
            )
            btn_f_deep.pack(side="left", padx=3)

        # Radio Effect Stage
        radio = self.engine.get_stage("radio_effect")
        if radio:
            rf = StageFrame(
                self,
                "Alastor Vintage Radio DSP",
                stage=radio,
                on_toggle_enable=lambda val: self.toggle_stage("radio_effect", val)
            )
            rf.pack(fill="x", padx=10, pady=6)
            self.frames["radio_effect"] = rf

            s_drive = SliderRow(
                rf.content,
                "Tube Overdrive",
                minimum=0.5,
                maximum=5.0,
                value=float(getattr(radio, "drive", 2.2)),
                callback=lambda val: setattr(radio, "drive", val)
            )
            s_drive.pack(fill="x", padx=5, pady=3)

            s_makeup = SliderRow(
                rf.content,
                "Makeup Gain",
                minimum=0.5,
                maximum=3.0,
                value=float(getattr(radio, "makeup", 1.4)),
                callback=lambda val: setattr(radio, "makeup", val)
            )
            s_makeup.pack(fill="x", padx=5, pady=3)

        # 1. RVC Voice Conversion
        rvc = self.engine.get_stage("rvc")
        if rvc:
            f = StageFrame(
                self,
                "RVC Voice Conversion (ROCm AMD GPU)",
                stage=rvc,
                on_toggle_enable=lambda val: self.toggle_stage("rvc", val)
            )
            f.pack(fill="x", padx=10, pady=6)
            self.frames["rvc"] = f

            # Model Selection Row
            model_row = ctk.CTkFrame(f.content, fg_color="transparent")
            model_row.pack(fill="x", padx=5, pady=(5, 3))

            m_lbl = ctk.CTkLabel(model_row, text="Voice Model:", width=110, anchor="w")
            m_lbl.pack(side="left", padx=5)

            discovered = discover_rvc_models()
            model_map = dict(discovered)

            cur_path = getattr(rvc, "path_pth", "")
            cur_key = None
            for k, p in model_map.items():
                if os.path.abspath(p) == os.path.abspath(cur_path):
                    cur_key = k
                    break
            if not cur_key and model_map:
                cur_key = list(model_map.keys())[0]

            info_var = ctk.StringVar()
            def update_info_text():
                model_name = os.path.basename(getattr(rvc, 'path_pth', 'N/A'))
                sr = getattr(rvc, 'sr', 32000)
                info_var.set(f"Model: {model_name}  |  Native SR: {sr} Hz  |  Output: 48000 Hz  |  AMD ROCm GPU")

            update_info_text()

            # Forward declaration container for pitch slider
            pitch_controls = {}

            def on_model_selected(selected_name):
                pth = model_map.get(selected_name)
                if pth and os.path.exists(pth):
                    self.engine.set_rvc_model(pth)
                    update_info_text()
                    ps = pitch_controls.get("slider")
                    if ps:
                        if "xiaomu" in selected_name.lower() or "female" in selected_name.lower():
                            ps.set_value(7.0)
                        elif "alastor" in selected_name.lower():
                            ps.set_value(0.0)

            model_keys = list(model_map.keys()) if model_map else ["None"]
            model_opt = ctk.CTkOptionMenu(
                model_row,
                values=model_keys,
                command=on_model_selected,
                width=180
            )
            if cur_key:
                model_opt.set(cur_key)
            model_opt.pack(side="left", padx=5)

            def on_browse_model():
                chosen = filedialog.askopenfilename(
                    title="Select RVC Model (.pth)",
                    filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")]
                )
                if chosen:
                    label = os.path.splitext(os.path.basename(chosen))[0]
                    model_map[label] = chosen
                    model_opt.configure(values=list(model_map.keys()))
                    model_opt.set(label)
                    on_model_selected(label)

            browse_btn = ctk.CTkButton(
                model_row,
                text="Browse...",
                width=80,
                command=on_browse_model
            )
            browse_btn.pack(side="left", padx=5)

            info_lbl = ctk.CTkLabel(f.content, textvariable=info_var, anchor="w", text_color="#38bdf8")
            info_lbl.pack(fill="x", padx=10, pady=(2, 4))

            current_pitch = getattr(rvc, "pitch", 0)
            pitch_slider = SliderRow(
                f.content,
                "Pitch Shift",
                minimum=-24.0,
                maximum=24.0,
                value=float(current_pitch),
                callback=lambda val: rvc.set_parameters(pitch=int(round(val)))
            )
            pitch_slider.pack(fill="x", padx=5, pady=3)
            pitch_controls["slider"] = pitch_slider

            # Quick Pitch Presets
            preset_row = ctk.CTkFrame(f.content, fg_color="transparent")
            preset_row.pack(fill="x", padx=5, pady=(0, 6))

            p_lbl = ctk.CTkLabel(preset_row, text="Presets:", width=70, anchor="w")
            p_lbl.pack(side="left", padx=5)

            btn_nat_female = ctk.CTkButton(
                preset_row,
                text="+7 (Female)",
                width=100,
                height=24,
                command=lambda: pitch_slider.set_value(7.0)
            )
            btn_nat_female.pack(side="left", padx=3)

            btn_high_female = ctk.CTkButton(
                preset_row,
                text="+12 (High)",
                width=85,
                height=24,
                command=lambda: pitch_slider.set_value(12.0)
            )
            btn_high_female.pack(side="left", padx=3)

            btn_neutral = ctk.CTkButton(
                preset_row,
                text="0 (Neutral)",
                width=85,
                height=24,
                command=lambda: pitch_slider.set_value(0.0)
            )
            btn_neutral.pack(side="left", padx=3)

            btn_deep = ctk.CTkButton(
                preset_row,
                text="-5 (Deep)",
                width=85,
                height=24,
                command=lambda: pitch_slider.set_value(-5.0)
            )
            btn_deep.pack(side="left", padx=3)



        # 2. Input Gain
        gain = self.engine.get_stage("gain")
        if gain:
            f = StageFrame(
                self,
                "Input Gain",
                stage=gain,
                on_toggle_enable=lambda val: self.toggle_stage("gain", val)
            )
            f.pack(fill="x", padx=10, pady=6)
            self.frames["gain"] = f

            s = SliderRow(
                f.content,
                "Gain",
                minimum=0.0,
                maximum=5.0,
                value=float(getattr(gain, "gain", 1.0)),
                callback=lambda val: gain.set_parameters(gain=val)
            )
            s.pack(fill="x", padx=5, pady=3)

        # 3. High-Pass Filter (HPF)
        hpf = self.engine.get_stage("high_pass")
        if hpf:
            f = StageFrame(
                self,
                "High-Pass Filter (HPF)",
                stage=hpf,
                on_toggle_enable=lambda val: self.toggle_stage("high_pass", val)
            )
            f.pack(fill="x", padx=10, pady=6)
            self.frames["high_pass"] = f

            s = SliderRow(
                f.content,
                "Cutoff (Hz)",
                minimum=20.0,
                maximum=500.0,
                value=float(getattr(hpf, "cutoff", 120.0)),
                callback=lambda val: hpf.set_parameters(cutoff=val)
            )
            s.pack(fill="x", padx=5, pady=3)

        # 4. Low-Pass Filter (LPF)
        lpf = self.engine.get_stage("low_pass")
        if lpf:
            f = StageFrame(
                self,
                "Low-Pass Filter (LPF)",
                stage=lpf,
                on_toggle_enable=lambda val: self.toggle_stage("low_pass", val)
            )
            f.pack(fill="x", padx=10, pady=6)
            self.frames["low_pass"] = f

            s = SliderRow(
                f.content,
                "Cutoff (Hz)",
                minimum=1000.0,
                maximum=20000.0,
                value=float(min(20000.0, getattr(lpf, "cutoff", 16000.0))),
                callback=lambda val: lpf.set_parameters(cutoff=val)
            )
            s.pack(fill="x", padx=5, pady=3)

        # 5. Noise Gate
        ng = self.engine.get_stage("noisegate")
        if ng:
            f = StageFrame(
                self,
                "Noise Gate",
                stage=ng,
                on_toggle_enable=lambda val: self.toggle_stage("noisegate", val)
            )
            f.pack(fill="x", padx=10, pady=6)
            self.frames["noisegate"] = f

            s_th = SliderRow(
                f.content,
                "Threshold",
                minimum=0.001,
                maximum=0.1,
                value=float(getattr(ng, "threshold", 0.035)),
                callback=lambda val: ng.set_parameters(threshold=val)
            )
            s_th.pack(fill="x", padx=5, pady=3)

            s_atk = SliderRow(
                f.content,
                "Attack (s)",
                minimum=0.001,
                maximum=0.1,
                value=float(getattr(ng, "attack", 0.015)),
                callback=lambda val: ng.set_parameters(atk=val)
            )
            s_atk.pack(fill="x", padx=5, pady=3)

            s_rls = SliderRow(
                f.content,
                "Release (s)",
                minimum=0.01,
                maximum=0.5,
                value=float(getattr(ng, "release", 0.125)),
                callback=lambda val: ng.set_parameters(rls=val)
            )
            s_rls.pack(fill="x", padx=5, pady=3)

        # 6. Compressor
        comp = self.engine.get_stage("Compressor")
        if comp:
            f = StageFrame(
                self,
                "Compressor",
                stage=comp,
                on_toggle_enable=lambda val: self.toggle_stage("Compressor", val)
            )
            f.pack(fill="x", padx=10, pady=6)
            self.frames["Compressor"] = f

            s_th = SliderRow(
                f.content,
                "Threshold",
                minimum=0.01,
                maximum=1.0,
                value=float(getattr(comp, "threshold", 0.2)),
                callback=lambda val: setattr(comp, "threshold", val)
            )
            s_th.pack(fill="x", padx=5, pady=3)

            s_ratio = SliderRow(
                f.content,
                "Ratio",
                minimum=1.0,
                maximum=30.0,
                value=float(getattr(comp, "ratiou", 14.0)),
                callback=lambda val: setattr(comp, "ratiou", val)
            )
            s_ratio.pack(fill="x", padx=5, pady=3)

        # 7. Limiter
        lim = self.engine.get_stage("limiter")
        if lim:
            f = StageFrame(
                self,
                "Limiter",
                stage=lim,
                on_toggle_enable=lambda val: self.toggle_stage("limiter", val)
            )
            f.pack(fill="x", padx=10, pady=6)
            self.frames["limiter"] = f

            s_th = SliderRow(
                f.content,
                "Threshold",
                minimum=0.1,
                maximum=1.0,
                value=float(getattr(lim, "threshold", 0.7)),
                callback=lambda val: setattr(lim, "threshold", val)
            )
            s_th.pack(fill="x", padx=5, pady=3)

            s_drive = SliderRow(
                f.content,
                "Drive",
                minimum=0.1,
                maximum=2.0,
                value=float(getattr(lim, "drive", 0.6)),
                callback=lambda val: setattr(lim, "drive", val)
            )
            s_drive.pack(fill="x", padx=5, pady=3)

            opt = OptionRow(
                f.content,
                "Mode",
                values=["soft", "Hard"],
                current=getattr(lim, "mode", "soft"),
                callback=lambda val: setattr(lim, "mode", val)
            )
            opt.pack(fill="x", padx=5, pady=3)

    def toggle_stage(self, name, enabled):
        stage = self.engine.get_stage(name)
        if stage:
            if enabled:
                self.engine.audioPipeline.enable_stage(stage)
            else:
                self.engine.audioPipeline.disable_stage(stage)

    def refresh(self):
        self.build()

    def update_values(self):
        pass
    