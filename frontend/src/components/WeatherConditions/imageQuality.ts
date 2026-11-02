/**
 * Error function, Abramowitz & Stegun 7.1.26 (max absolute error 1.5e-7).
 */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

/**
 * Percentile (0-100) of on-source image quality.
 *
 * Port of the "qap" model in lucupy's `Conditions.iq_to_percentile`
 * (lucupy/minimodel/constraints.py), fit to QAP data from 2004-2024.
 * Keep the coefficients in sync with it.
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
  // The extra +0.5 is to force 100% in the worst IQ
  const c = [
    50.10221383, 0.87712202, 0.78467697, 16.10928544, 0.13778389, -15.8255612,
    49.37405633 + 0.5,
  ];
  const p =
    c[0] *
      erf(c[1] * wavelength ** c[2] + c[3] * zenithFwhm ** c[4] + c[5]) +
    c[6];
  return Math.max(0, Math.min(100, p));
}
