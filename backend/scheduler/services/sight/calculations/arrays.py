import numpy as np
import numpy.typing as npt


# =============================================================================
# Core Pack/Unpack
# =============================================================================

def pack_array(arr: npt.NDArray[np.float64]) -> bytes:
    """
    Pack a float64 numpy array into bytes for database storage.
    
    Args:
        arr: 1D numpy array of float64
        
    Returns:
        bytes representation of the array
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got {arr.ndim}D")
    return arr.tobytes()


def unpack_array(data: bytes, length: int | None = None) -> npt.NDArray[np.float64]:
    """
    Unpack bytes into a float64 numpy array.
    
    Args:
        data: bytes from database
        length: expected length (optional, for validation)
        
    Returns:
        1D numpy array of float64
    """
    arr = np.frombuffer(data, dtype=np.float64)
    if length is not None and len(arr) != length:
        raise ValueError(f"Expected length {length}, got {len(arr)}")
    return arr


# =============================================================================
# Binary Mask Pack/Unpack (for Stage 2 visibility results)
# =============================================================================

def pack_binary_mask(mask: npt.NDArray[np.bool_] | npt.NDArray[np.uint8]) -> bytes:
    """
    Pack a binary mask into bytes using bit-packing (8x compression).
    
    Args:
        mask: 1D array of booleans or 0/1 integers
        
    Returns:
        Packed bytes (length = ceil(len(mask) / 8))
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D array, got {mask.ndim}D")
    return np.packbits(mask).tobytes()


def unpack_binary_mask(data: bytes, length: int) -> npt.NDArray[np.bool_]:
    """
    Unpack bytes into a binary mask.
    
    Args:
        data: Packed bytes from database
        length: Original array length (needed because packbits pads to multiple of 8)
        
    Returns:
        1D boolean numpy array
    """
    packed = np.frombuffer(data, dtype=np.uint8)
    unpacked = np.unpackbits(packed)[:length]
    return unpacked.astype(np.bool_)


# =============================================================================
# Angle Conversions
# =============================================================================

def radians_to_degrees(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert radians to degrees."""
    return np.degrees(arr)


def degrees_to_radians(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert degrees to radians."""
    return np.radians(arr)


# =============================================================================
# Convenience Classes
# =============================================================================

class ArrayPacker:
    """
    Convenience class for packing multiple arrays into a dict of bytes.
    
    Usage:
        packer = ArrayPacker()
        packed = packer.pack_all(
            alt=alt_array,
            az=az_array,
            airmass=airmass_array,
        )
        # packed = {'alt': b'...', 'az': b'...', 'airmass': b'...'}
    """
    
    @staticmethod
    def pack_all(**arrays: npt.NDArray[np.float64]) -> dict[str, bytes]:
        """Pack multiple named arrays."""
        return {name: pack_array(arr) for name, arr in arrays.items()}
    
    @staticmethod
    def unpack_all(data: dict[str, bytes], length: int) -> dict[str, npt.NDArray[np.float64]]:
        """Unpack multiple named arrays."""
        return {name: unpack_array(arr_bytes, length) for name, arr_bytes in data.items()}


class AngularArrayPacker:
    """
    Packer that handles radian storage with optional degree conversion on read.
    
    Usage:
        packer = AngularArrayPacker()
        
        # Store (converts to radians if input is degrees)
        packed = packer.pack(alt_degrees, input_unit='degrees')
        
        # Retrieve (converts to requested unit)
        alt_degrees = packer.unpack(packed, length=490, output_unit='degrees')
        alt_radians = packer.unpack(packed, length=490, output_unit='radians')
    """
    
    @staticmethod
    def pack(
        arr: npt.NDArray[np.float64], 
        input_unit: str = 'radians'
    ) -> bytes:
        """
        Pack angular array, storing in radians.
        
        Args:
            arr: Angular values
            input_unit: 'radians' or 'degrees'
        """
        if input_unit == 'degrees':
            arr = degrees_to_radians(arr)
        elif input_unit != 'radians':
            raise ValueError(f"Unknown unit: {input_unit}")
        return pack_array(arr)
    
    @staticmethod
    def unpack(
        data: bytes, 
        length: int,
        output_unit: str = 'radians'
    ) -> npt.NDArray[np.float64]:
        """
        Unpack angular array from radians storage.
        
        Args:
            data: Packed bytes
            length: Array length
            output_unit: 'radians' or 'degrees'
        """
        arr = unpack_array(data, length)
        if output_unit == 'degrees':
            return radians_to_degrees(arr)
        elif output_unit == 'radians':
            return arr
        else:
            raise ValueError(f"Unknown unit: {output_unit}")


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_night_duration(duration: int, min_dur: int = 60, max_dur: int = 720) -> None:
    """
    Validate night duration is within reasonable bounds.
    
    Args:
        duration: Night duration in minutes
        min_dur: Minimum valid duration
        max_dur: Maximum valid duration
        
    Raises:
        ValueError: If duration is outside bounds
    """
    if not min_dur <= duration <= max_dur:
        raise ValueError(
            f"Night duration {duration} outside valid range [{min_dur}, {max_dur}]"
        )


def validate_array_length(arr: npt.NDArray, expected: int, name: str = "array") -> None:
    """
    Validate array has expected length.
    
    Args:
        arr: Array to validate
        expected: Expected length
        name: Name for error message
        
    Raises:
        ValueError: If length doesn't match
    """
    if len(arr) != expected:
        raise ValueError(f"{name} length {len(arr)} != expected {expected}")