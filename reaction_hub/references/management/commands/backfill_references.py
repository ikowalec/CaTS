from __future__ import annotations

from django.core.management.base import BaseCommand

from references.models import CalculationSettingsProfile, ReferenceAdsorbate, ReferenceNanocluster


class Command(BaseCommand):
    help = "Create baseline reference adsorbate/nanocluster/settings entries if missing."

    def handle(self, *args, **options):
        profile, _ = CalculationSettingsProfile.objects.get_or_create(
            name="default_pbe0_water",
            defaults={
                "approved": True,
                "xc_functional": "PBE0",
                "basis_set": "6-31g(d,p)",
                "solvent": "Water",
                "integration_grid": "DefGrid2",
                "scf_convergence": "TightSCF",
                "metadata": {"source": "bootstrap"},
            },
        )
        nwchem_profile, _ = CalculationSettingsProfile.objects.get_or_create(
            name="nwchem_pbe0_def2svp",
            defaults={
                "approved": True,
                "xc_functional": "PBE0",
                "basis_set": "def2-SVP",
                "solvent": "",
                "integration_grid": "fine",
                "scf_convergence": "TightSCF",
                "metadata": {
                    "source": "bootstrap",
                    "qm_backend": "nwchem",
                    "calculator": "nwchem",
                    "nwchem": {
                        "version": "7.2.2",
                        "memory_mb": 1000,
                        "molden": True,
                    },
                },
                "notes": "Default NWChem settings for online test submissions.",
            },
        )
        adsorbate, _ = ReferenceAdsorbate.objects.get_or_create(
            name="O2",
            defaults={
                "formula": "O2",
                "charge": 0,
                "multiplicity": 3,
                "settings_profile": profile,
                "notes": "Seed adsorbate for adsorption-energy consistency checks.",
            },
        )
        cluster, _ = ReferenceNanocluster.objects.get_or_create(
            name="Cu13",
            defaults={
                "composition": "Cu13",
                "atom_count": 13,
                "charge": 0,
                "multiplicity": 1,
                "settings_profile": profile,
                "notes": "Seed nanocluster for adsorption-energy consistency checks.",
            },
        )

        self.stdout.write(
            self.style.SUCCESS(
                "Backfill complete: "
                f"profile={profile.name}, nwchem_profile={nwchem_profile.name}, "
                f"adsorbate={adsorbate.name}, cluster={cluster.name}"
            )
        )
