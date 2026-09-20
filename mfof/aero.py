"""Aerodynamic evaluator for the MFOF framework.

Thin alias around :class:`liu2019.aero.Liu2019AeroEvaluator`. The Liu
evaluator is fully duck-typed on the waverider object (it only calls
``wr.upper_surface(mirror=True)`` and ``wr.lower_surface(mirror=True)``), so
:class:`MFOFWaverider` works in place of :class:`Liu2019Waverider` without
modification.

The underlying evaluator offers three local-inclination pressure laws --
``"cone"`` (tangent-cone, the default), ``"oblique"`` (tangent-wedge) and
``"newtonian"`` -- selectable per evaluator or per call. See the
:mod:`liu2019.aero` module docstring for how they compare.

Note the default suits a cone-derived waverider. An MFOF waverider built
with the wedge or power-law factory has a different generating body, so
``solver="oblique"`` is the better match for an all-wedge build.
"""

from liu2019.aero import Liu2019AeroEvaluator


class MFOFAeroEvaluator(Liu2019AeroEvaluator):
    """Local-inclination aero evaluator for an MFOF waverider.

    Constructor and public methods are inherited verbatim:

    * ``__init__(waverider, gamma=1.4, ref_length=6.0, ref_area=1.0,
                  moment_ref=(0,0,0), solver="cone")``
    * ``evaluate(Ma, alpha_deg=0.0, atm_conditions=None, solver=None)``
    * ``evaluate_paper_trajectory(progress_callback=None, solver=None)``
    * ``compare_with_paper(solver=None)``
    """
    pass
