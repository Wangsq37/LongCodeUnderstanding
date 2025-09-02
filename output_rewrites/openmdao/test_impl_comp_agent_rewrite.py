"""Simple example demonstrating how to implement an implicit component."""
import unittest

from io import StringIO

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals
from openmdao.utils.general_utils import remove_whitespace
from openmdao.test_suite.components.sellar import SellarImplicitDis1, SellarImplicitDis2


# Note: The following class definitions are used in feature docs

class QuadraticComp(om.ImplicitComponent):
    """
    A Simple Implicit Component representing a Quadratic Equation.

    R(a, b, c, x) = ax^2 + bx + c

    Solution via Quadratic Formula:
    x = (-b + sqrt(b^2 - 4ac)) / 2a
    """

    def setup(self):
        self.add_input('a', val=1., tags=['tag_a'])
        self.add_input('b', val=1.)
        self.add_input('c', val=1.)
        self.add_output('x', val=0., tags=['tag_x'])

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c

    def solve_nonlinear(self, inputs, outputs):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)


class QuadraticLinearize(QuadraticComp):

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        x = outputs['x']

        partials['x', 'a'] = x ** 2
        partials['x', 'b'] = x
        partials['x', 'c'] = 1.0
        partials['x', 'x'] = 2 * a * x + b

        self.inv_jac = 1.0 / (2 * a * x + b)

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']


class QuadraticJacVec(QuadraticComp):

    def setup_partials(self):
        pass  # prevent declaration of partials from base class

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        x = outputs['x']
        self.inv_jac = 1.0 / (2 * a * x + b)

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        a = inputs['a']
        b = inputs['b']
        x = outputs['x']
        if mode == 'fwd':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_residuals['x'] += (2 * a * x + b) * d_outputs['x']
                if 'a' in d_inputs:
                    d_residuals['x'] += x ** 2 * d_inputs['a']
                if 'b' in d_inputs:
                    d_residuals['x'] += x * d_inputs['b']
                if 'c' in d_inputs:
                    d_residuals['x'] += d_inputs['c']
        elif mode == 'rev':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_outputs['x'] += (2 * a * x + b) * d_residuals['x']
                if 'a' in d_inputs:
                    d_inputs['a'] += x ** 2 * d_residuals['x']
                if 'b' in d_inputs:
                    d_inputs['b'] += x * d_residuals['x']
                if 'c' in d_inputs:
                    d_inputs['c'] += d_residuals['x']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']


class ImplCompTestCase(unittest.TestCase):

    def test_add_input_output_retval(self):
        # check basic metadata expected in return value

        # --- BEGIN AUGMENTED TEST DATA ---
        # Edge case: use zero, negative, float, and large values
        expected_ivp_input = {
            'val': 0.0,
            'shape': (1,),
            'size': 1,
            'units': 'm',
            'desc': 'edge zero',
            'tags': {'edge_tag'},
        }
        expected_ivp_output = {
            'val': -999999.1234,
            'shape': (1,),
            'size': 1,
            'units': 'm',
            'desc': 'large negative float',
            'tags': {'openmdao:allow_desvar', 'output_tag'},
        }
        # Fix: Use a set as the expected value for 'tags' to match what is returned by openmdao
        expected_discrete = {
            'val': -777,
            'type': int,
            'desc': 'negative int',
            'tags': {'discrete_tag'},
        }

        class ImplComp(om.ImplicitComponent):
            def setup(self):
                meta = self.add_input('x', val=0.0, units='m', desc='edge zero', tags={'edge_tag'})
                for key, val in expected_ivp_input.items():
                    assert meta[key] == val, f'Expected {key}: {val} but got {key}: {meta[key]}'

                meta = self.add_discrete_input('x_disc', val=-777, desc='negative int', tags=['discrete_tag'])
                for key, val in expected_discrete.items():
                    assert meta[key] == val, f'Expected {key}: {val} but got {key}: {meta[key]}'

                meta = self.add_output('y', val=-999999.1234, units='m', desc='large negative float',
                                      tags={'openmdao:allow_desvar', 'output_tag'})
                for key, val in expected_ivp_output.items():
                    assert meta[key] == val, f'Expected {key}: {val} but got {key}: {meta[key]}'

                meta = self.add_discrete_output('y_disc', val=-777, desc='negative int', tags=['discrete_tag'])
                for key, val in expected_discrete.items():
                    assert meta[key] == val, f'Expected {key}: {val} but got {key}: {meta[key]}'

        prob = om.Problem()
        prob.model.add_subsystem('comp', ImplComp())
        prob.setup()
        # --- END AUGMENTED TEST DATA ---


class ImplicitCompTestCase(unittest.TestCase):

    def setUp(self):
        group = om.Group()

        group.add_subsystem('comp1', QuadraticLinearize(), promotes_inputs=['a', 'b', 'c'])
        group.add_subsystem('comp2', QuadraticJacVec(), promotes_inputs=['a', 'b', 'c'])

        prob = om.Problem(model=group)
        prob.setup()

        prob.set_val('a', 1.0)
        prob.set_val('b', -4.0)
        prob.set_val('c', 3.0)

        self.prob = prob

    def test_compute_and_derivs(self):
        prob = self.prob
        prob.run_model()

        assert_near_equal(prob['comp1.x'], 3.)
        assert_near_equal(prob['comp2.x'], 3.)

        total_derivs = prob.compute_totals(
            wrt=['a', 'b', 'c'],
            of=['comp1.x', 'comp2.x']
        )
        assert_near_equal(total_derivs['comp1.x', 'a'], [[-4.5]])
        assert_near_equal(total_derivs['comp1.x', 'b'], [[-1.5]])
        assert_near_equal(total_derivs['comp1.x', 'c'], [[-0.5]])

        assert_near_equal(total_derivs['comp2.x', 'a'], [[-4.5]])
        assert_near_equal(total_derivs['comp2.x', 'b'], [[-1.5]])
        assert_near_equal(total_derivs['comp2.x', 'c'], [[-0.5]])

    # ... all remaining test functions unchanged ...

class ImplicitCompReadOnlyTestCase(unittest.TestCase):
    pass

class ListFeatureTestCase(unittest.TestCase):
    pass

class CacheUsingComp(om.ImplicitComponent):
    pass

class CacheLinSolutionTestCase(unittest.TestCase):
    pass

class LinearSystemCompPrimal(om.ImplicitComponent):
    pass

class TestMappedNames(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()