# interaction_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading
import time
import json
from typing import Optional
from enum import Enum

from interaction.kws import KWSModule
from interaction.ptt import PTTModule
from drivers.bluetooth import BluetoothModule
from drivers.audio import AudioModule
from drivers.button import ButtonModule
from audio.wakeup import EnergyVAD
from audio.preprocess import PreprocessModule
from audio.asr import ASRModule
from audio.nlu import NLUModule
from audio.tts import TTSModule

class SystemMode(Enum):
    IDLE = 1
    NAVIGATION = 2
    POWER_OFF = 3

class _KwsModelStub:
    def detect(self, frame_bytes: bytes) -> Optional[str]:
        return None

class _AsrModelStub:
    def SenseVoiceSmall_ASR(self, path):
        return ("ok", "离线识别测试文本")


class InteractionNode(Node):
    TTS_COOLDOWN_SEC = 0.5

    def __init__(self):
        # ★ 修复 1：必须初始化父类
        super().__init__("interaction_node")

        # ---------------- 参数配置 ----------------
        param_names = [
            "sample_rate", "mic_index", "frame_duration_ms", "max_silence_frames",
            "vad_threshold", "record_timeout", "bt_target_mac", "button_pin",
            "button_pull_up", "button_debounce_ms", "button_poll_interval_ms",
            "ptt_record_file_base", "ptt_a2dp_profile", "ptt_hsp_profile",
            "ptt_auto_switch_profile", "ptt_max_keep_files", "ptt_poll_interval",
            "ptt_record_max_duration", "keywords", "enable_denoiser",
            "use_online_asr", "nlu_wakeup_words", "nlu_custom_rules",
            "enable_tts", "tts_model", "tts_json"
        ]
        for name in param_names:
            # 给个安全默认值，防止 yaml 没配置时崩溃
            self.declare_parameter(name, "") 
        self.cfg = {name: self.get_parameter(name).value for name in param_names}

        # ---------------- ROS 发布器与订阅器 ----------------
        self.text_pub = self.create_publisher(String, "/interaction/asr_text", 10)
        self.wakeup_pub = self.create_publisher(Bool, "/interaction/wakeup", 10)
        self.intent_pub = self.create_publisher(String, "/interaction/intent", 10)
        self.status_pub = self.create_publisher(String, "/interaction/status", 10)
        
        self.guidance_sub = self.create_subscription(
            String, "/navigation/guidance", self._on_navigation_guidance, 10
        )

        # ---------------- 节点状态 ----------------
        self._state_lock = threading.Lock()
        self._threads_lock = threading.Lock()
        self._running = False
        self._last_tts_end_time = 0.0
        # ★ 修复 2：统一变量名
        self._system_mode = SystemMode.IDLE 

        # ---------------- 创建模块 ----------------
        self._create_modules(self.cfg)

        self.modules = [
            self.bluetooth, self.audio, self.button, self.preprocess,
            self.kws, self.asr, self.nlu,
        ]
        if self.tts is not None:
            self.modules.append(self.tts)
        self.modules.append(self.ptt)

        self.loops = [
            (self.bluetooth, self.bluetooth.stdout_loop),
            (self.bluetooth, self.bluetooth.stderr_loop),
            (self.bluetooth, self.bluetooth.reconnect_loop),
            (self.audio, self.audio.audio_loop),
            (self.button, self.button.button_loop),
            (self.kws, self.kws.kws_loop),
            (self.asr, self.asr.asr_loop),
        ]
        if self.tts is not None:
            self.loops.append((self.tts, self.tts.tts_loop))
        self.loops.append((self.ptt, self.ptt.ptt_loop))

        self._threads = []
        self.timer = self.create_timer(0.1, self._poll_results)

        self.get_logger().info(f"Interaction node initialized (TTS={'on' if self.tts else 'off'})")

    # ... (省略 _create_modules 和 _parse_custom_rules 保持你的原样) ...
    def _create_modules(self, cfg):
        # 你的原版代码，保持不变
        self.bluetooth = BluetoothModule({"target_mac": cfg["bt_target_mac"]})
        self.audio = AudioModule(cfg={}, bt_manager=self.bluetooth)
        self.button = ButtonModule({"pin": cfg["button_pin"], "pull_up": cfg["button_pull_up"], "debounce_ms": cfg["button_debounce_ms"], "poll_interval_ms": cfg["button_poll_interval_ms"]})
        self.vad = EnergyVAD(threshold=cfg["vad_threshold"])
        self.preprocess = PreprocessModule({"enable": cfg["enable_denoiser"], "sample_rate": cfg["sample_rate"]})
        self.kws = KWSModule({"sample_rate": cfg["sample_rate"], "mic_index": cfg["mic_index"], "frame_duration_ms": cfg["frame_duration_ms"], "max_silence_frames": cfg["max_silence_frames"], "keywords": cfg["keywords"], "kws_model": _KwsModelStub(), "audio_path": "/tmp/interaction_kws_utterance.wav", "record_timeout": cfg["record_timeout"]}, vad=self.vad)
        self.asr = ASRModule({"use_online_asr": cfg["use_online_asr"]}, model_interface=_AsrModelStub(), kws=self.kws, preprocess=self.preprocess)
        self.nlu = NLUModule({"wakeup_words": cfg["nlu_wakeup_words"], "custom_rules": self._parse_custom_rules(cfg["nlu_custom_rules"])})
        self.tts = None
        if cfg["enable_tts"] and cfg["tts_model"] and cfg["tts_json"]:
            self.tts = TTSModule({"zh_tts_model": cfg["tts_model"], "zh_tts_json": cfg["tts_json"]}, logger=self.get_logger(), audio_manager=self.audio)
        self.ptt = PTTModule({"record_file_base": cfg["ptt_record_file_base"], "a2dp_profile": cfg["ptt_a2dp_profile"], "hsp_profile": cfg["ptt_hsp_profile"], "auto_switch_profile": cfg["ptt_auto_switch_profile"], "max_keep_files": cfg["ptt_max_keep_files"], "poll_interval": cfg["ptt_poll_interval"], "record_max_duration": cfg["ptt_record_max_duration"]}, asr=self.asr, audio=self.audio, button=self.button)

    def _parse_custom_rules(self, raw) -> dict:
        if not raw: return {}
        if isinstance(raw, dict): return raw
        try: return json.loads(raw)
        except Exception: return {}

    # ================================================================
    #   生命周期 & 状态机
    # ================================================================
    def start(self):
        failed_modules = set()
        for m in self.modules:
            try: m.start()
            except Exception as e:
                self.get_logger().error(f"Module {m.__class__.__name__} start failed: {e}")
                failed_modules.add(m)

        for module, loop_func in self.loops:
            if module in failed_modules: continue
            t = threading.Thread(target=loop_func, name=f"{module.__class__.__name__}", daemon=True)
            t.start()
            with self._threads_lock: self._threads.append(t)

        with self._state_lock: self._running = True
        
        # 启动后先更新状态，再触发播报
        self._system_mode = SystemMode.IDLE
        self._speak("系统已就绪")
        
        self._publish_status({"event": "started"})
        self.get_logger().info("Interaction node started")

    def set_mode(self, new_mode: SystemMode):
        if self._system_mode == new_mode:
            return

        self.get_logger().info(f"Switching mode: {self._system_mode.name} -> {new_mode.name}")
        
        # 1. 离开当前状态的清理
        if self._system_mode == SystemMode.IDLE:
            self.kws.deactivate()  
        elif self._system_mode == SystemMode.NAVIGATION:
            if self.tts: self.tts.stop()

        # 2. 更新状态
        self._system_mode = new_mode

        # 3. 进入新状态的硬件切换 (注意：这里不负责播报，避免逻辑冲突)
        if self._system_mode == SystemMode.IDLE:
            try:
                self.kws.activate()    
            except Exception as e:
                self.get_logger().error(f"KWS activate failed: {e}")
            
        elif self._system_mode == SystemMode.NAVIGATION:
            pass # KWS 保持关闭
            
        elif self._system_mode == SystemMode.POWER_OFF:
            if self.tts: self.tts.stop()
            self.get_logger().info("System entering POWER OFF mode.")

    def shutdown(self):
        with self._state_lock: self._running = False
        
        # ★ 安全销毁定时器
        if hasattr(self, 'timer') and self.timer is not None:
            self.destroy_timer(self.timer)
            self.timer = None

        try: self.kws.deactivate()
        except Exception: pass
        try: self.audio.stop_audio()
        except Exception: pass

        for m in self.modules:
            try: m.shutdown()
            except Exception as e: self.get_logger().error(f"{m.__class__.__name__} shutdown failed: {e}")

        self._publish_status({"event": "shutdown"})

    def cleanup(self):
        for m in reversed(self.modules):
            try: m.cleanup()
            except Exception: pass

        with self._threads_lock:
            threads = list(self._threads)
            self._threads.clear()
        for t in threads:
            t.join(timeout=2.0)

    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    # ================================================================
    #   ROS 定时器轮询
    # ================================================================
    def _poll_results(self):
        if not self.is_running(): return
        for fn in (self._poll_wakeup_event, self._poll_asr_result, self._poll_tts_state):
            try: fn()
            except Exception as e: self.get_logger().error(f"{fn.__name__} failed: {e}")

    def _poll_wakeup_event(self):
        if not self.kws.is_keyword_detected(): return
        keyword = self.kws.get_detected_keyword()
        self.kws.clear_detection()

        if self._system_mode != SystemMode.IDLE: return 

        msg = Bool()
        msg.data = True
        self.wakeup_pub.publish(msg)
        self._publish_status({"event": "wakeup", "keyword": keyword})

    def _poll_asr_result(self):
        text = self.asr.get_result()
        if not text: return

        msg = String()
        msg.data = text
        self.text_pub.publish(msg)
        self._publish_status({"event": "asr", "text": text})

        nlu_result = self.nlu.parse(text)
        self._publish_status({"event": "nlu", "result": nlu_result})
        self._dispatch(nlu_result)

    def _poll_tts_state(self):
        if self.tts is None: return

        # ★ 修复 5：导航模式下 TTS 播报时不理会 KWS（因为 KWS 本来就关了）。
        # 只要处于 IDLE，就执行防自触发。
        if self._system_mode == SystemMode.IDLE:
            busy = self.tts.is_busy()
            now = time.time()
            if busy:
                self._last_tts_end_time = now
                if not self.kws.is_paused(): self.kws.pause()
            else:
                if self.kws.is_paused() and (now - self._last_tts_end_time) > self.TTS_COOLDOWN_SEC:
                    self.kws.resume()

    # ================================================================
    #   意图分发
    # ================================================================
    def _dispatch(self, nlu_result: dict):
        intent = nlu_result.get("intent", "unknown")
        argument = nlu_result.get("argument", "")

        intent_msg = String()
        intent_msg.data = json.dumps(nlu_result, ensure_ascii=False)
        self.intent_pub.publish(intent_msg)

        if intent == "navigate":
            if self._system_mode != SystemMode.NAVIGATION:
                self.set_mode(SystemMode.NAVIGATION) # 此处已不再自带播报
            response = f"正在为您导航到{argument}" if argument else "请告诉我目的地"
            self._speak(response)

        elif intent == "stop":
            if self._system_mode != SystemMode.IDLE:
                self.set_mode(SystemMode.IDLE) # 此处已不再自带播报
            self._speak("已停止导航")

        elif intent == "power_off":
            self.set_mode(SystemMode.POWER_OFF)

        elif intent == "describe_scene":
            self._speak("正在为您描述场景")
        elif intent == "read_text":
            self._speak("正在为您朗读")
        else:
            if self._system_mode == SystemMode.IDLE:
                self._speak("抱歉，我没有听懂")

    def _on_navigation_guidance(self, msg: String):
        if self._system_mode == SystemMode.NAVIGATION:
            text = msg.data
            self.get_logger().info(f"Received guidance: {text}")
            self._speak(text)

    def _speak(self, text: str):
        if self.tts is None: return
        
        # ★ 修复 4：如果在 IDLE 模式播报，暂停 KWS
        if self._system_mode == SystemMode.IDLE:
            self.kws.pause()
            self._last_tts_end_time = time.time()
            
        try:
            self.tts.speak(text, wait_finish=False)
        except Exception as e:
            self.get_logger().error(f"TTS speak failed: {e}")
            if self._system_mode == SystemMode.IDLE: self.kws.resume()

    def _publish_status(self, status: dict):
        status["mode"] = self._system_mode.name 
        msg = String()
        msg.data = json.dumps(status, ensure_ascii=False)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode()
    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()