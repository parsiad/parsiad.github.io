# %% [raw]
# +++
# aliases = [
#   "/blog/2024/an_introduction_to_the_curiously_recurring_template_pattern"
# ]
# date = 2024-09-11
# title = "An introduction to the curiously repeating template pattern"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# ## Motivation
#
# The curiously recurring template pattern (CRTP) is a C++ polymorphism technique allowing a base class to access derived class members without virtual table lookups.
# This is accomplished via [template metaprogramming](https://en.wikipedia.org/wiki/Template_metaprogramming).
#
# Note that aggressive applications of template metaprogramming can often be a form of premature optimization, reducing the readability of a code base.
# We do *not* claim that CRTP should be used whenever possible but rather that it *can* be useful in certain contexts.
# For example, [the Eigen library](https://eigen.tuxfamily.org) uses CRTP to implement [expression templates](https://en.wikipedia.org/wiki/Expression_templates) in order to optimize its usage.
#
# In this short post, we demonstrate how to use CRTP to remove occurrences of the virtual keyword from a toy example.
#
# ## Virtual polymorphism
#
# Consider the following abstract class:
#
# ```c++
# class Animal {
#   string name;
#   virtual const vector<string> &get_sounds() const = 0;
#
# public:
#   Animal(const string &name) : name(name) {}
#   string get_random_sound(mt19937 &gen) const {
#     const auto &sounds = this->get_sounds();
#     uniform_int_distribution<> dist(0, sounds.size() - 1);
#     return sounds[dist(gen)];
#   }
#   string get_name() const { return name; }
# };
# ```
#
# Its job is simple: whenever `get_random_sound` is called, the animal produces one of a finite number of sounds.
# The sound is chosen randomly from a list of sounds the animal is able to make.
#
# Lets add some animals:
#
# ```c++
# class Cat : public Animal {
#   const vector<string> &get_sounds() const override {
#     static const vector<string> sounds = {"hiss", "meow"};
#     return sounds;
#   }
#
# public:
#   Cat(const string &name) : Animal(name) {}
# };
#
# class Dog : public Animal {
#   const vector<string> &get_sounds() const override {
#     static const vector<string> sounds = {"bark", "growl", "whine"};
#     return sounds;
#   }
#
# public:
#   Dog(const string &name) : Animal(name) {}
# };
# ```
#
# Finally, let's have some animals produce some sounds:
#
# ```c++
# int main() {
#   random_device rd;
#   mt19937 gen(rd());
#
#   vector<unique_ptr<Animal>> animals;
#   animals.emplace_back(make_unique<Cat>("Charlie"));
#   animals.emplace_back(make_unique<Dog>("Daisy"));
#
#   for (auto &animal : animals) {
#     cout << animal->get_name() << ": " << animal->get_random_sound(gen) << endl;
#   }
#
#   return 0;
# }
# ```
#
# Running this program, we might see output such as the following:
#
# ```
# Charlie: meow
# Daisy: whine
# ```
#
# ## CRTP polymorphism
#
# Now, let's translate our implementation to CRTP.
# The main idea in CRTP is to template the base class as `Base<Derived>` (in our case, the base class is `Animal`) in order to access the members of the derived class from the base class code.
# Below is our revised base class.
#
# ```c++
# template <typename Derived> class Animal {
#   string name;
#
# public:
#   Animal(const string &name) : name(name) {}
#   string get_name() const { return name; }
#   string get_random_sound(mt19937 &gen) const {
#     const auto &sounds = static_cast<const Derived &>(*this).get_sounds();
#     uniform_int_distribution<> dist(0, sounds.size() - 1);
#     return sounds[dist(gen)];
#   }
# };
# ```
#
# Casts of the form `static_cast<const Derived &>(*this)` are common to CRTP code: this is because `this` is a pointer to the base class and we need to be able to treat it like a pointer to the derived class in order to access members of the derived class.
#
# Next, let's port our derived classes:
#
# ```c++
# class Cat : public Animal<Cat> {
# public:
#   static const vector<string> &get_sounds() {
#     static const vector<string> sounds = {"hiss", "meow"};
#     return sounds;
#   }
#
# public:
#   Cat(const string &name) : Animal(name) {}
# };
#
# class Dog : public Animal<Dog> {
# public:
#   static const vector<string> &get_sounds() {
#     static const vector<string> sounds = {"bark", "growl", "whine"};
#     return sounds;
#   }
#
# public:
#   Dog(const string &name) : Animal(name) {}
# };
# ```
#
# To avoid virtual table lookups, we use [`variant`](https://en.cppreference.com/w/cpp/utility/variant) and [`visit`](https://en.cppreference.com/w/cpp/utility/variant/visit) to iterate through the animals:
#
# ```c++
# int main() {
#   random_device rd;
#   mt19937 gen(rd());
#
#   vector<variant<Cat, Dog>> animals;
#   animals.emplace_back(Cat("Charlie"));
#   animals.emplace_back(Dog("Daisy"));
#
#   for (const auto &animal : animals) {
#     visit(
#         [&gen](const auto &a) {
#           cout << a.get_name() << ": " << a.get_random_sound(gen) << endl;
#         },
#         animal);
#   }
#
#   return 0;
# }
# ```
#
# ## Further reading
#
# While the above example is useful in introducing CRTP, the interested reader is encouraged to read about [expression templates](https://en.wikipedia.org/wiki/Expression_templates) to see a more realistic application of CRTP.
