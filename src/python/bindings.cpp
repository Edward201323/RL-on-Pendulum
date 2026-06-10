#include <pybind11/pybind11.h>

#include "physics/cartpole.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cartpole_cpp, m) {
    m.doc() = "Headless cart-pole physics core (shared with the SFML visualizer).";

    py::class_<CartPoleConfig>(m, "CartPoleConfig")
        .def(py::init<>())
        .def_readwrite("gravity", &CartPoleConfig::gravity)
        .def_readwrite("cart_mass", &CartPoleConfig::cartMass)
        .def_readwrite("bob_mass", &CartPoleConfig::bobMass)
        .def_readwrite("length", &CartPoleConfig::length)
        .def_readwrite("pole_damping", &CartPoleConfig::poleDamping)
        .def_readwrite("cart_friction", &CartPoleConfig::cartFriction)
        .def_readwrite("track_limit", &CartPoleConfig::trackLimit)
        .def_readwrite("initial_angle", &CartPoleConfig::initialAngle)
        .def_readwrite("fail_angle", &CartPoleConfig::failAngle)
        .def_readwrite("max_dt", &CartPoleConfig::maxDt)
        .def_readwrite("substeps", &CartPoleConfig::substeps);

    py::class_<CartPole>(m, "CartPole")
        .def(py::init<>())
        .def(py::init<const CartPoleConfig&>(), py::arg("config"))
        .def("reset", &CartPole::reset)
        .def("set_state", &CartPole::setState, py::arg("x"), py::arg("velocity"),
             py::arg("theta"), py::arg("angular_velocity"))
        .def("set_control_force", &CartPole::setControlForce, py::arg("force"))
        .def("get_control_force", &CartPole::getControlForce)
        .def("advance", &CartPole::advance, py::arg("dt"))
        .def("is_done", &CartPole::isDone)
        // -1 left wall, +1 right wall, 0 free; lets the env penalize wall hits.
        .def("boundary_contact", &CartPole::boundaryContact)
        .def_property_readonly("x", &CartPole::getX)
        .def_property_readonly("velocity", &CartPole::getVelocity)
        .def_property_readonly("angle", &CartPole::getAngle)
        .def_property_readonly("angular_velocity", &CartPole::getAngularVelocity)
        // reference_internal: the returned config is owned by the CartPole, so
        // mutating it (curriculum learning) writes straight back into the sim.
        .def_property_readonly("config", &CartPole::config,
                               py::return_value_policy::reference_internal);
}
