//! A layer's parameters by their paths, as PEFT's and torch's state dictionaries name them:
//! `layers.0.q_proj.adapters.0.a`. A struct says which of its fields hold parameters with
//! [`module_parameters!`](crate::module_parameters), each field a [`Param`] or another layer, or a
//! `Vec` of them, whose elements are numbered in the path.

use super::array::Array;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;

/// Gradients or parameters by their paths.
pub type FlattenedModuleParam = HashMap<Rc<str>, Array>;

/// A layer: a forward pass, over parameters.
pub trait Module<Input>: ModuleParameters + std::fmt::Debug {
    /// What the forward pass returns.
    type Output;
    /// How it fails.
    type Error: std::error::Error;

    /// The forward pass.
    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error>;

    /// Sets training or evaluation mode, for what differs between them.
    fn training_mode(&mut self, mode: bool);
}

/// A parameter: its value, and whether it is frozen, so left out of what trains.
#[derive(Debug, Clone)]
pub struct Param<T> {
    /// The value.
    pub value: T,
    frozen: bool,
}

impl<T> Param<T> {
    /// A trainable parameter.
    pub fn new(value: T) -> Self {
        Self {
            value,
            frozen: false,
        }
    }

    /// Leaves it out of what trains.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Whether it is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
}

impl<T> Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> AsRef<T> for Param<T> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

/// Something holding parameters at paths: a [`Param`], a layer or a `Vec` of either.
pub trait ModuleParameters {
    /// Calls `f` with each parameter, its path `prefix` and beyond.
    fn visit<'a>(&'a self, prefix: &str, f: &mut dyn FnMut(String, &'a Param<Array>));

    /// Calls `f` with each parameter to change, its path `prefix` and beyond.
    fn visit_mut<'a>(&'a mut self, prefix: &str, f: &mut dyn FnMut(String, &'a mut Param<Array>));

    /// Every parameter by its path, frozen ones included.
    fn parameters(&self) -> HashMap<Rc<str>, &Array> {
        let mut parameters = HashMap::new();
        self.visit("", &mut |path, param| {
            parameters.insert(path.into(), &param.value);
        });
        parameters
    }

    /// Every parameter not frozen, by its path.
    fn trainable_parameters(&self) -> HashMap<Rc<str>, &Array> {
        let mut parameters = HashMap::new();
        self.visit("", &mut |path, param| {
            if !param.is_frozen() {
                parameters.insert(path.into(), &param.value);
            }
        });
        parameters
    }

    /// Replaces the parameters at these paths; a path this does not hold is ignored.
    fn update_flattened(&mut self, mut parameters: FlattenedModuleParam) {
        self.visit_mut("", &mut |path, param| {
            if let Some(value) = parameters.remove(path.as_str()) {
                param.value = value;
            }
        });
    }

    /// Freezes every parameter.
    fn freeze_parameters(&mut self) {
        self.visit_mut("", &mut |_, param| param.freeze());
    }
}

/// `prefix.name`, or `name` where the prefix is empty; [`module_parameters!`](crate::module_parameters)'s.
#[doc(hidden)]
pub fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

impl ModuleParameters for Param<Array> {
    fn visit<'a>(&'a self, prefix: &str, f: &mut dyn FnMut(String, &'a Param<Array>)) {
        f(prefix.to_owned(), self);
    }

    fn visit_mut<'a>(&'a mut self, prefix: &str, f: &mut dyn FnMut(String, &'a mut Param<Array>)) {
        f(prefix.to_owned(), self);
    }
}

impl<T: ModuleParameters> ModuleParameters for Vec<T> {
    fn visit<'a>(&'a self, prefix: &str, f: &mut dyn FnMut(String, &'a Param<Array>)) {
        for (i, item) in self.iter().enumerate() {
            item.visit(&join(prefix, &i.to_string()), f);
        }
    }

    fn visit_mut<'a>(&'a mut self, prefix: &str, f: &mut dyn FnMut(String, &'a mut Param<Array>)) {
        for (i, item) in self.iter_mut().enumerate() {
            item.visit_mut(&join(prefix, &i.to_string()), f);
        }
    }
}

/// Implements [`ModuleParameters`] for a struct from the
/// fields that hold its parameters: `module_parameters!(Block { up, gate, down });`.
#[macro_export]
macro_rules! module_parameters {
    ($t:ty { $($field:ident),* $(,)? }) => {
        impl $crate::mlx::module::ModuleParameters for $t {
            fn visit<'a>(
                &'a self,
                prefix: &str,
                f: &mut dyn FnMut(String, &'a $crate::mlx::module::Param<$crate::mlx::Array>),
            ) {
                $(
                    $crate::mlx::module::ModuleParameters::visit(
                        &self.$field,
                        &$crate::mlx::module::join(prefix, stringify!($field)),
                        f,
                    );
                )*
            }

            fn visit_mut<'a>(
                &'a mut self,
                prefix: &str,
                f: &mut dyn FnMut(String, &'a mut $crate::mlx::module::Param<$crate::mlx::Array>),
            ) {
                $(
                    $crate::mlx::module::ModuleParameters::visit_mut(
                        &mut self.$field,
                        &$crate::mlx::module::join(prefix, stringify!($field)),
                        f,
                    );
                )*
            }
        }
    };
}
