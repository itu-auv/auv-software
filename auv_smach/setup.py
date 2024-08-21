from setuptools import setup, find_packages

setup(
    name="auv_smach",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["smach", "smach_ros"],
    entry_points={
        "console_scripts": [
            "main_state_machine = auv_smach.main_state_machine:main",
        ],
    },
)
