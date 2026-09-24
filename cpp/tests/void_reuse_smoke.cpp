#include "batched.hpp"
#include "soma.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

// setenv/unsetenv are POSIX; the Microsoft C runtime has _putenv_s (an empty value unsets).
static void set_env(const char *name,const char *value) {
#ifdef _WIN32
  _putenv_s(name,value?value:"");
#else
  if(value) setenv(name,value,1); else unsetenv(name);
#endif
}

namespace {
void require(bool value,const char *message) {if(!value)throw std::runtime_error(message);}
template<class T> void same(const std::vector<T>&a,const std::vector<T>&b){
  require(a.size()==b.size(),"vector size mismatch");
  require(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0,"vector bytes mismatch");
}
void same(const brook::Array&a,const brook::Array&b){
  require(a.shape==b.shape&&a.dtype==b.dtype,"array metadata mismatch");
  std::vector<unsigned char>x(a.bytes()),y(b.bytes());
  a.copy_to_host(x.data(),x.size());b.copy_to_host(y.data(),y.size());same(x,y);
}
brook::SomaPreparation prepare(brook::Context &ctx,const brook::Components &cc,const brook::Array &dbf,
    const std::vector<brook::Box>&boxes,bool fixing,bool reuse){
  set_env("BROOK_VOID_REUSE",reuse?nullptr:"0");
  brook::SkeletonizeOptions options;options.dust=50;options.fix_branching=fixing;options.fix_borders=false;
  options.teasar.scale=1.5;options.teasar.constant=2;options.teasar.pdrf_scale=100000;options.teasar.pdrf_exponent=4;
  options.teasar.soma_detection=.5;options.teasar.soma_acceptance=2;options.teasar.soma_scale=1;options.teasar.max_paths=8;
  std::vector<std::vector<brook::Point>> empty(cc.mapping.size());
  auto fields=brook::prepare_batched(ctx,cc,dbf,boxes,empty,options);
  require(fields.filled[1]>0&&fields.filled[2]>0,"both nested shells must need filling");
  std::weak_ptr<brook::Allocation> retained_storage;
  if(reuse&&!fields.filled_masks.empty())retained_storage=fields.filled_masks.begin()->second.storage;
  size_t retained=0;for(auto &entry:fields.filled_masks)retained+=entry.second.bytes();
  if(reuse){
    require(fields.filled_masks.size()==1,"second mask must exceed retention budget");
    require(retained==44*44*44,"wrong retained mask");
    require(retained<=cc.labels.size(),"retention budget exceeded");
  }else require(retained==0,"disabled retention must own no masks");
  auto out=brook::prepare_somata(ctx,cc,dbf,std::move(fields),boxes,empty,empty,empty,options);
  require(out.fields.filled_masks.empty(),"retained masks escaped preparation phase");
  require(retained_storage.expired(),"retained mask allocation survived preparation");
  require(out.layout.dimensions.size()==2,"outer shell must use a private arena");
  return out;
}
}
int main(){
  set_env("BROOK_SOMA_EVICT","0");
  constexpr int side=48;std::vector<uint16_t> image(side*side*side,0);
  auto cube=[&](int lo,int hi,uint16_t label){for(int z=lo;z<hi;++z)for(int y=lo;y<hi;++y)for(int x=lo;x<hi;++x)image[x+side*(y+side*z)]=label;};
  cube(2,46,3);cube(3,45,0);cube(4,44,5);cube(5,43,0);
  brook::Context ctx(0);brook_volume input{};input.struct_size=sizeof(input);input.abi_version=BROOK_ABI_VERSION;
  input.data=image.data();input.dtype=BROOK_U16;input.memory=BROOK_HOST;
  for(int a=0;a<3;++a){input.shape[a]=side;input.strides[a]=2*(a==0?1:a==1?side:side*side);}
  auto labels=brook::upload(ctx,input);auto cc=brook::connected_components(ctx,labels);require(cc.mapping.size()==3,"expected two components");
  auto boxes=brook::analyze(ctx,cc.labels,cc.roots.size());auto dbf=brook::edt(ctx,cc.labels,{1,1,1},false);
  for(bool fixing:{false,true}){
    auto old=prepare(ctx,cc,dbf,boxes,fixing,false),retained=prepare(ctx,cc,dbf,boxes,fixing,true);
    same(old.components.labels,retained.components.labels);same(old.dbf,retained.dbf);
    same(old.fields.daf,retained.fields.daf);same(old.fields.pdrf,retained.fields.pdrf);same(old.fields.parents,retained.fields.parents);
    same(old.fields.active,retained.fields.active);same(old.fields.soma,retained.fields.soma);same(old.fields.root,retained.fields.root);same(old.fields.target,retained.fields.target);
    same(old.fields.dbf_max,retained.fields.dbf_max);same(old.fields.max_daf,retained.fields.max_daf);
  }
  std::cout<<"PASS: nested-shell retention budget, uncached fallback, private arena, phase release and exact prepared fields in both modes\n";
}
