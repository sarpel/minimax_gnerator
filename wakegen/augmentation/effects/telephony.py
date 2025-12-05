"""
Telephony Simulation Module

This module provides realistic telephony and VoIP audio effects.
It simulates various phone call conditions, codecs, and network effects.

Key Features:
- Standard telephone bandwidth (PSTN, landline)
- Mobile phone codecs (AMR, AMR-WB)
- VoIP codec simulation (Opus, G.711, G.722)
- Network jitter and packet loss
- GSM codec artifacts

Think of this as a "phone call simulator" - it takes clean audio and
makes it sound like it came through a phone call, whether that's
an old landline, a cell phone, or a Zoom call.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from wakegen.core.exceptions import AugmentationError
from wakegen.utils.logging import get_logger

logger = get_logger("wakegen.augmentation.telephony")


class PhoneType(Enum):
    """Types of telephone systems to simulate."""
    PSTN_LANDLINE = "pstn_landline"      # Traditional landline (300-3400Hz)
    MOBILE_GSM = "mobile_gsm"             # GSM mobile (AMR codec)
    MOBILE_HD = "mobile_hd"               # HD Voice (AMR-WB codec)
    VOIP_LOW = "voip_low"                 # Low quality VoIP (Opus 8kbps)
    VOIP_MED = "voip_medium"              # Medium quality VoIP (Opus 16kbps)
    VOIP_HIGH = "voip_high"               # High quality VoIP (Opus 32kbps)
    CORDLESS_DECT = "cordless_dect"       # DECT cordless phone


@dataclass
class TelephonyConfig:
    """
    Configuration for telephony simulation.
    
    Attributes:
        phone_type: Type of phone system to simulate
        packet_loss_rate: Probability of packet loss (0.0 to 0.1)
        jitter_ms: Network jitter in milliseconds
        echo_level: Echo level (0.0 = none, 0.3 = noticeable)
        background_noise_level: Level of background noise
    """
    phone_type: PhoneType = PhoneType.VOIP_MED
    packet_loss_rate: float = 0.02
    jitter_ms: float = 20.0
    echo_level: float = 0.0
    background_noise_level: float = 0.01


class TelephonySimulator:
    """
    Simulates various telephony conditions and codecs.
    
    This class applies realistic phone call effects to audio,
    making clean recordings sound like they came through different
    types of phone systems.
    
    Usage:
        sim = TelephonySimulator(sample_rate=16000)
        
        # Simulate a mobile phone call
        processed = sim.simulate_phone_call(audio, PhoneType.MOBILE_GSM)
        
        # Full simulation with network effects
        config = TelephonyConfig(
            phone_type=PhoneType.VOIP_MED,
            packet_loss_rate=0.03,
            jitter_ms=30.0
        )
        processed = sim.apply_full_simulation(audio, config)
    """
    
    # Frequency characteristics for each phone type
    PHONE_CHARACTERISTICS: Dict[PhoneType, Dict] = {
        PhoneType.PSTN_LANDLINE: {
            "low_cut": 300,
            "high_cut": 3400,
            "sample_rate": 8000,
            "description": "Traditional landline phone",
        },
        PhoneType.MOBILE_GSM: {
            "low_cut": 200,
            "high_cut": 3400,
            "sample_rate": 8000,
            "quantization_bits": 13,
            "description": "GSM mobile phone (2G/3G)",
        },
        PhoneType.MOBILE_HD: {
            "low_cut": 50,
            "high_cut": 7000,
            "sample_rate": 16000,
            "description": "HD Voice mobile (VoLTE)",
        },
        PhoneType.VOIP_LOW: {
            "low_cut": 200,
            "high_cut": 4000,
            "sample_rate": 8000,
            "description": "Low quality VoIP (Opus 8kbps)",
        },
        PhoneType.VOIP_MED: {
            "low_cut": 50,
            "high_cut": 8000,
            "sample_rate": 16000,
            "description": "Medium quality VoIP (Opus 16kbps)",
        },
        PhoneType.VOIP_HIGH: {
            "low_cut": 20,
            "high_cut": 12000,
            "sample_rate": 24000,
            "description": "High quality VoIP (Opus 32kbps)",
        },
        PhoneType.CORDLESS_DECT: {
            "low_cut": 150,
            "high_cut": 6800,
            "sample_rate": 16000,
            "description": "DECT cordless phone",
        },
    }
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the telephony simulator.
        
        Args:
            sample_rate: Target sample rate for processing.
        """
        self.sample_rate = sample_rate
    
    def apply_bandpass_filter(
        self,
        audio: np.ndarray,
        low_cut: float,
        high_cut: float
    ) -> np.ndarray:
        """
        Apply a bandpass filter to simulate phone bandwidth.
        
        Args:
            audio: Input audio signal.
            low_cut: Low frequency cutoff in Hz.
            high_cut: High frequency cutoff in Hz.
        
        Returns:
            Filtered audio signal.
        """
        try:
            # Use FFT-based filtering for efficiency
            fft_audio = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
            
            # Create smooth rolloff filter (not brick-wall)
            # This is more realistic for phone systems
            filter_response = np.ones(len(freqs))
            
            # Low-frequency rolloff
            low_rolloff = 1.0 / (1.0 + (low_cut / (freqs + 1e-6)) ** 2)
            
            # High-frequency rolloff
            high_rolloff = 1.0 / (1.0 + (freqs / high_cut) ** 4)
            
            filter_response = low_rolloff * high_rolloff
            
            # Apply filter
            filtered = np.fft.irfft(fft_audio * filter_response, n=len(audio))
            
            return filtered.astype(audio.dtype)
        
        except Exception as e:
            raise AugmentationError(f"Bandpass filter failed: {e}") from e
    
    def add_codec_artifacts(
        self,
        audio: np.ndarray,
        quantization_bits: int = 16,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Add codec-like artifacts (quantization noise, compression effects).
        
        Args:
            audio: Input audio signal.
            quantization_bits: Simulated codec bit depth.
            add_noise: Whether to add codec noise floor.
        
        Returns:
            Audio with codec artifacts.
        """
        try:
            # Quantize to simulate codec bit depth
            scale = 2 ** (quantization_bits - 1)
            quantized = np.round(audio * scale) / scale
            
            # Add slight noise floor typical of speech codecs
            if add_noise:
                noise_level = 10 ** (-60 / 20)  # -60dB noise floor
                noise = np.random.randn(len(quantized)) * noise_level
                quantized = quantized + noise
            
            return np.clip(quantized, -1.0, 1.0)
        
        except Exception as e:
            raise AugmentationError(f"Codec artifacts failed: {e}") from e
    
    def add_packet_loss(
        self,
        audio: np.ndarray,
        loss_rate: float = 0.02,
        packet_size_ms: float = 20.0
    ) -> np.ndarray:
        """
        Simulate network packet loss by zeroing out random segments.
        
        Args:
            audio: Input audio signal.
            loss_rate: Probability of packet loss (0.0 to 0.1).
            packet_size_ms: Typical packet size in milliseconds.
        
        Returns:
            Audio with simulated packet loss.
        """
        try:
            result = audio.copy()
            packet_samples = int(packet_size_ms * self.sample_rate / 1000)
            
            # Process audio in packets
            for i in range(0, len(result), packet_samples):
                if np.random.random() < loss_rate:
                    # Simulate packet loss with concealment
                    # (simple method: fade to zero and back)
                    end_idx = min(i + packet_samples, len(result))
                    fade_len = min(50, (end_idx - i) // 2)
                    
                    # Create fade out and fade in
                    if fade_len > 0:
                        fade_out = np.linspace(1, 0, fade_len)
                        fade_in = np.linspace(0, 1, fade_len)
                        
                        result[i:i+fade_len] *= fade_out
                        result[end_idx-fade_len:end_idx] *= fade_in
                        result[i+fade_len:end_idx-fade_len] = 0
            
            return result
        
        except Exception as e:
            raise AugmentationError(f"Packet loss simulation failed: {e}") from e
    
    def add_jitter(
        self,
        audio: np.ndarray,
        jitter_ms: float = 20.0
    ) -> np.ndarray:
        """
        Simulate network jitter by applying slight time stretching variations.
        
        Args:
            audio: Input audio signal.
            jitter_ms: Jitter amount in milliseconds.
        
        Returns:
            Audio with jitter effects.
        """
        try:
            # Simplified jitter simulation - add micro-variations
            # Real jitter is more complex but this captures the effect
            
            jitter_samples = int(jitter_ms * self.sample_rate / 1000)
            
            # Add slight random time shifts to segments
            result = audio.copy()
            segment_size = self.sample_rate // 50  # 20ms segments
            
            for i in range(0, len(result) - segment_size, segment_size):
                shift = np.random.randint(-jitter_samples // 4, jitter_samples // 4)
                if shift > 0 and i + segment_size + shift < len(result):
                    # Small interpolation to smooth the shift
                    result[i:i+segment_size] = result[i+shift:i+segment_size+shift]
            
            return result
        
        except Exception as e:
            raise AugmentationError(f"Jitter simulation failed: {e}") from e
    
    def add_phone_echo(
        self,
        audio: np.ndarray,
        echo_delay_ms: float = 150.0,
        echo_level: float = 0.1
    ) -> np.ndarray:
        """
        Add telephone echo effect.
        
        Args:
            audio: Input audio signal.
            echo_delay_ms: Echo delay in milliseconds.
            echo_level: Echo amplitude (0.0 to 0.3).
        
        Returns:
            Audio with echo.
        """
        try:
            delay_samples = int(echo_delay_ms * self.sample_rate / 1000)
            
            result = audio.copy()
            
            # Add delayed echo
            for i in range(delay_samples, len(result)):
                result[i] += audio[i - delay_samples] * echo_level
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result /= max_val
            
            return result
        
        except Exception as e:
            raise AugmentationError(f"Echo simulation failed: {e}") from e
    
    def simulate_phone_call(
        self,
        audio: np.ndarray,
        phone_type: PhoneType
    ) -> np.ndarray:
        """
        Apply phone-specific frequency and codec characteristics.
        
        Args:
            audio: Input audio signal.
            phone_type: Type of phone to simulate.
        
        Returns:
            Processed audio sounding like a phone call.
        """
        try:
            characteristics = self.PHONE_CHARACTERISTICS[phone_type]
            
            # Apply bandpass filter
            processed = self.apply_bandpass_filter(
                audio,
                low_cut=characteristics["low_cut"],
                high_cut=characteristics["high_cut"]
            )
            
            # Add codec quantization if specified
            if "quantization_bits" in characteristics:
                processed = self.add_codec_artifacts(
                    processed,
                    quantization_bits=characteristics["quantization_bits"]
                )
            else:
                processed = self.add_codec_artifacts(processed)
            
            logger.debug(f"Applied {phone_type.value} simulation")
            return processed
        
        except Exception as e:
            raise AugmentationError(f"Phone call simulation failed: {e}") from e
    
    def apply_full_simulation(
        self,
        audio: np.ndarray,
        config: TelephonyConfig
    ) -> np.ndarray:
        """
        Apply complete telephony simulation including network effects.
        
        Args:
            audio: Input audio signal.
            config: Telephony configuration.
        
        Returns:
            Fully processed audio.
        """
        try:
            # Apply phone characteristics
            processed = self.simulate_phone_call(audio, config.phone_type)
            
            # Apply network effects
            if config.packet_loss_rate > 0:
                processed = self.add_packet_loss(processed, config.packet_loss_rate)
            
            if config.jitter_ms > 0:
                processed = self.add_jitter(processed, config.jitter_ms)
            
            if config.echo_level > 0:
                processed = self.add_phone_echo(processed, echo_level=config.echo_level)
            
            # Add background noise
            if config.background_noise_level > 0:
                noise = np.random.randn(len(processed)) * config.background_noise_level
                processed = processed + noise
                processed = np.clip(processed, -1.0, 1.0)
            
            return processed
        
        except Exception as e:
            raise AugmentationError(f"Full telephony simulation failed: {e}") from e
    
    def get_available_phone_types(self) -> Dict[str, str]:
        """Get list of available phone types with descriptions."""
        return {
            pt.value: self.PHONE_CHARACTERISTICS[pt]["description"]
            for pt in PhoneType
        }


# =============================================================================
# DISTANCE SIMULATION
# =============================================================================


@dataclass
class DistanceConfig:
    """
    Configuration for distance/position simulation.
    
    Attributes:
        distance_meters: Distance from microphone in meters
        room_size: Small/medium/large room
        air_absorption: Whether to simulate high-frequency air absorption
        direct_to_reverb_ratio: Ratio of direct sound to reverb
    """
    distance_meters: float = 1.0
    room_size: str = "medium"  # small, medium, large
    air_absorption: bool = True
    direct_to_reverb_ratio: float = 0.7


class DistanceSimulator:
    """
    Simulates the effect of speaking at different distances from a microphone.
    
    As distance increases:
    - Volume decreases (inverse square law)
    - High frequencies attenuate more (air absorption)
    - More room reverb is picked up
    - Signal becomes "thinner" due to proximity effect loss
    
    This is useful for training wake word models that need to work
    at various distances.
    
    Usage:
        sim = DistanceSimulator(sample_rate=16000)
        
        # Simulate speaking from 3 meters away
        processed = sim.simulate_distance(audio, distance_meters=3.0)
        
        # Full simulation with room characteristics
        config = DistanceConfig(distance_meters=2.0, room_size="large")
        processed = sim.apply_full_simulation(audio, config)
    """
    
    # Room reverb characteristics
    ROOM_CHARACTERISTICS = {
        "small": {"rt60": 0.3, "size_factor": 1.0},   # Small room/closet
        "medium": {"rt60": 0.5, "size_factor": 2.0},  # Normal room
        "large": {"rt60": 0.8, "size_factor": 4.0},   # Large room/hall
        "outdoor": {"rt60": 0.1, "size_factor": 10.0}, # Outdoor (minimal reverb)
    }
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the distance simulator.
        
        Args:
            sample_rate: Target sample rate for processing.
        """
        self.sample_rate = sample_rate
    
    def apply_distance_attenuation(
        self,
        audio: np.ndarray,
        distance_meters: float,
        reference_distance: float = 0.3
    ) -> np.ndarray:
        """
        Apply volume attenuation based on distance (inverse square law).
        
        Args:
            audio: Input audio signal.
            distance_meters: Distance from source in meters.
            reference_distance: Reference distance (close-mic position).
        
        Returns:
            Attenuated audio.
        """
        try:
            # Inverse square law: level decreases with square of distance
            # With some practical limits
            distance = max(reference_distance, distance_meters)
            attenuation = (reference_distance / distance) ** 1.5  # 1.5 for indoor
            attenuation = max(0.01, min(1.0, attenuation))  # Clamp
            
            return audio * attenuation
        
        except Exception as e:
            raise AugmentationError(f"Distance attenuation failed: {e}") from e
    
    def apply_air_absorption(
        self,
        audio: np.ndarray,
        distance_meters: float
    ) -> np.ndarray:
        """
        Simulate high-frequency absorption through air.
        
        Air absorbs high frequencies more than low frequencies,
        especially over longer distances.
        
        Args:
            audio: Input audio signal.
            distance_meters: Distance in meters.
        
        Returns:
            Audio with high-frequency attenuation.
        """
        try:
            # Air absorption increases with frequency and distance
            # Simplified model: apply gentle lowpass with distance
            
            if distance_meters < 1.0:
                return audio  # Minimal effect at close range
            
            # Calculate cutoff frequency based on distance
            # At 1m: 16kHz, at 10m: ~6kHz (simplified model)
            base_cutoff = 16000
            cutoff = base_cutoff / (1 + 0.5 * (distance_meters - 1))
            cutoff = max(3000, min(cutoff, self.sample_rate / 2 - 100))
            
            # Apply lowpass filter
            fft_audio = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
            
            # Gentle rolloff
            filter_response = 1.0 / (1.0 + (freqs / cutoff) ** 2)
            filtered = np.fft.irfft(fft_audio * filter_response, n=len(audio))
            
            return filtered.astype(audio.dtype)
        
        except Exception as e:
            raise AugmentationError(f"Air absorption simulation failed: {e}") from e
    
    def apply_proximity_effect_loss(
        self,
        audio: np.ndarray,
        distance_meters: float
    ) -> np.ndarray:
        """
        Simulate loss of proximity effect at increased distances.
        
        Proximity effect boosts bass when speaking close to a mic.
        At distance, this boost is lost, making the voice "thinner".
        
        Args:
            audio: Input audio signal.
            distance_meters: Distance in meters.
        
        Returns:
            Audio with proximity effect loss.
        """
        try:
            if distance_meters < 0.3:
                # Add proximity effect (bass boost) for very close mic
                fft_audio = np.fft.rfft(audio)
                freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
                
                # Boost low frequencies
                boost = 1.0 + 0.3 / (1.0 + (freqs / 200) ** 2)
                boosted = np.fft.irfft(fft_audio * boost, n=len(audio))
                return boosted.astype(audio.dtype)
            
            elif distance_meters > 1.0:
                # Reduce bass for distant source
                fft_audio = np.fft.rfft(audio)
                freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
                
                # Gentle high-pass effect
                factor = min(1.0, (distance_meters - 1.0) * 0.2)
                reduction = 1.0 - factor / (1.0 + (freqs / 150) ** 2)
                filtered = np.fft.irfft(fft_audio * reduction, n=len(audio))
                return filtered.astype(audio.dtype)
            
            return audio
        
        except Exception as e:
            raise AugmentationError(f"Proximity effect simulation failed: {e}") from e
    
    def add_simple_reverb(
        self,
        audio: np.ndarray,
        rt60: float = 0.5,
        wet_level: float = 0.3
    ) -> np.ndarray:
        """
        Add simple reverb to simulate room acoustics.
        
        Args:
            audio: Input audio signal.
            rt60: Reverb time (time for sound to decay 60dB).
            wet_level: Amount of reverb (0.0 to 1.0).
        
        Returns:
            Audio with reverb.
        """
        try:
            # Simple reverb using multiple delays
            # This is a very simplified model; real reverb is more complex
            
            reverb = np.zeros(len(audio) + int(rt60 * self.sample_rate))
            reverb[:len(audio)] = audio
            
            # Add multiple delayed copies with decay
            delays_ms = [23, 37, 53, 71, 97, 131, 173, 227]
            
            for delay_ms in delays_ms:
                delay_samples = int(delay_ms * self.sample_rate / 1000)
                decay = np.exp(-3 * delay_ms / (rt60 * 1000))
                
                for i in range(delay_samples, len(reverb)):
                    if i - delay_samples < len(audio):
                        reverb[i] += audio[i - delay_samples] * decay * 0.3
            
            # Trim to original length
            reverb = reverb[:len(audio)]
            
            # Mix dry and wet
            result = audio * (1 - wet_level) + reverb * wet_level
            
            # Normalize
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result /= max_val
            
            return result
        
        except Exception as e:
            raise AugmentationError(f"Reverb simulation failed: {e}") from e
    
    def simulate_distance(
        self,
        audio: np.ndarray,
        distance_meters: float
    ) -> np.ndarray:
        """
        Apply basic distance simulation.
        
        Args:
            audio: Input audio signal.
            distance_meters: Distance from microphone.
        
        Returns:
            Distance-simulated audio.
        """
        try:
            # Apply distance effects
            processed = self.apply_distance_attenuation(audio, distance_meters)
            processed = self.apply_air_absorption(processed, distance_meters)
            processed = self.apply_proximity_effect_loss(processed, distance_meters)
            
            return processed
        
        except Exception as e:
            raise AugmentationError(f"Distance simulation failed: {e}") from e
    
    def apply_full_simulation(
        self,
        audio: np.ndarray,
        config: DistanceConfig
    ) -> np.ndarray:
        """
        Apply complete distance and room simulation.
        
        Args:
            audio: Input audio signal.
            config: Distance configuration.
        
        Returns:
            Fully processed audio.
        """
        try:
            # Get room characteristics
            room = self.ROOM_CHARACTERISTICS.get(
                config.room_size,
                self.ROOM_CHARACTERISTICS["medium"]
            )
            
            # Apply distance effects
            processed = self.simulate_distance(audio, config.distance_meters)
            
            # Apply air absorption if enabled
            if config.air_absorption:
                processed = self.apply_air_absorption(processed, config.distance_meters)
            
            # Calculate reverb amount based on distance and room
            # More reverb at greater distances
            reverb_amount = min(0.7, config.distance_meters * 0.1 * room["size_factor"])
            reverb_amount *= (1 - config.direct_to_reverb_ratio)
            
            if reverb_amount > 0.05:
                processed = self.add_simple_reverb(
                    processed,
                    rt60=room["rt60"],
                    wet_level=reverb_amount
                )
            
            return processed
        
        except Exception as e:
            raise AugmentationError(f"Full distance simulation failed: {e}") from e
    
    def get_available_room_sizes(self) -> Dict[str, str]:
        """Get available room size presets with descriptions."""
        return {
            "small": "Small room/closet (RT60: 0.3s)",
            "medium": "Normal room (RT60: 0.5s)",
            "large": "Large room/hall (RT60: 0.8s)",
            "outdoor": "Outdoor/open space (minimal reverb)",
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Telephony
    "PhoneType",
    "TelephonyConfig",
    "TelephonySimulator",
    # Distance
    "DistanceConfig",
    "DistanceSimulator",
]
