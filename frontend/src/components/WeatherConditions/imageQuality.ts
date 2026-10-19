/**
 * Percentile (0-100) of on-source image quality.
 *
 * Port of the "web" model in lucupy's `Conditions.iq_to_percentile`
 * (lucupy/minimodel/constraints.py). Keep the coefficients in sync with it.
 *
 * @param fwhm on-source full width at half maximum, in arcsec
 * @param wavelength in micrometers
 * @param airmass airmass of the observation
 */
export function iqToPercentile(
  fwhm: number,
  wavelength: number,
  airmass: number
): number {
  const zenithFwhm = fwhm / airmass ** 0.6;
  const p =
    226.439637 -
    93.6673981 * wavelength ** -0.162045293 -
    47.4550797 * zenithFwhm ** -0.973500276;
  return Math.max(0, Math.min(100, p));
}
